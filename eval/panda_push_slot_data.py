"""Generate label-free visual/action data for the Panda slot-dynamics representation.

The dataset is deliberately separate from the 56-state monitor DEV panel.  It contains only RGB
sequences, executed end-effector actions, and train/validation pair membership.  Physical state,
contacts, success, toppling, monitor decisions and object identities are never written.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model, preprocess_frames
from jenga_short_held_tails import extract_sim
from panda_push_dev_panel import segment
from systems.panda_block_push import PUSH_XML, PandaBlockPush

OUT = ROOT / "results/panda_block_push/slot_dynamics"
RAW = OUT / "unlabeled_sequences.npz"
TOKENS = OUT / "unlabeled_tokens.npz"
DATA_MANIFEST = OUT / "data_manifest.json"
SEED = 640
TRAIN_PAIRS, VALIDATION_PAIRS = 64, 16
PAIRS = TRAIN_PAIRS + VALIDATION_PAIRS
ACTION_STEPS, HOLD_STEPS = 10, 6
HEIGHT = WIDTH = 160
TOKEN_DIM = 64
PROJECTION_SEED = 641
SHAPES = ((.0125, .0100, .0375), (.0180, .0140, .0275), (.0200, .0200, .0200))

PROTOCOL = {
    "id": "panda-generic-unlabeled-slot-dynamics-train-v0",
    "seed": SEED,
    "pairs": {"train": TRAIN_PAIRS, "validation": VALIDATION_PAIRS},
    "sequence": {"action_steps": ACTION_STEPS, "held_steps": HOLD_STEPS,
                 "frames_include_common_start": True},
    "render_size": [HEIGHT, WIDTH],
    "variation": ["three unlabeled cuboid geometries", "block pose", "push direction/height",
                  "camera position", "light intensity", "object colour"],
    "pairing": "same start and base action; symmetric nearby xyz action offsets <=3 mm",
    "stored_fields": ["frames", "actions", "split", "pair_index"],
    "excluded": ["physical state", "contact", "success", "topple", "object identity",
                 "physical monitor decision", "alarm"],
    "encoder": "frozen DINOv2-S/14 x_norm_patchtokens",
    "token_adapter": {"projection": "fixed seeded Gaussian", "seed": PROJECTION_SEED,
                      "dimensions": TOKEN_DIM},
    "resource_policy": "one simulator and one DINO batch at a time; BLAS <=2 threads",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_specs(seed=SEED, pairs=PAIRS):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(pairs):
        direction = rng.normal(size=2); direction /= max(np.linalg.norm(direction), 1e-8)
        length = rng.uniform(.035, .135)
        local = rng.normal(size=(ACTION_STEPS, 3))
        local = np.cumsum(local, axis=0)
        local -= local[:1]
        local /= max(float(np.max(np.linalg.norm(local, axis=1))), 1e-8)
        amplitude = rng.uniform(.0008, .0030)
        rows.append({
            "index": index, "shape": index % len(SHAPES),
            "block_xy": [rng.uniform(.515, .555), rng.uniform(-.055, .055)],
            "block_yaw": rng.uniform(-.55, .55), "contact_z": rng.uniform(.421, .457),
            "contact_offset_y": rng.uniform(-.020, .020),
            "direction": direction.tolist(), "length": float(length),
            "local_offset": (amplitude * local).tolist(),
            "camera_delta": rng.uniform([-.035, -.025, -.025], [.035, .025, .025]).tolist(),
            "light_scale": rng.uniform(.75, 1.25),
            "colour": rng.uniform([.25, .12, .08], [.95, .70, .35]).tolist(),
        })
    return rows


def _shape_xml(asset_directory, shape):
    original = 'size="0.0125 0.01 0.0375" type="box"'
    replacement = f'size="{shape[0]} {shape[1]} {shape[2]}" type="box"'
    text = PUSH_XML.replace(original, replacement)
    path = Path(asset_directory) / f"panda_slot_shape_{shape[0]:.4f}.xml"
    path.write_text(text)
    return path


def _configure_appearance(env, spec, nominal_camera, nominal_light):
    camera = env.model.camera("cam_fixed").id
    env.model.cam_pos[camera] = nominal_camera + np.asarray(spec["camera_delta"])
    env.model.light_diffuse[:] = nominal_light * float(spec["light_scale"])
    colour = np.asarray(spec["colour"], float)
    env.model.geom_rgba[env.block_geom_id, :3] = colour


def _render_pair(env, spec, nominal_camera, nominal_light):
    _configure_appearance(env, spec, nominal_camera, nominal_light)
    block_xy = np.asarray(spec["block_xy"], float)
    env.reset(block_xy=block_xy, block_yaw=float(spec["block_yaw"]))
    start = env.proprio()[:3]
    direction = np.asarray(spec["direction"], float)
    behind = np.array([block_xy[0] - .080, block_xy[1] + spec["contact_offset_y"],
                       spec["contact_z"]])
    contact = behind.copy(); contact[0] = block_xy[0] - .030
    for action in np.concatenate([segment(start, behind, 12), segment(behind, contact, 10)]):
        env.step(action)
    snapshot = env.snapshot()
    stop = contact.copy(); stop[:2] += float(spec["length"]) * direction
    base = segment(contact, stop, ACTION_STEPS)
    offset = np.asarray(spec["local_offset"], float)
    outputs = []
    for sign in (-1., 1.):
        env.restore(snapshot)
        chunk = base.copy(); chunk[:, :3] += sign * offset
        actions = np.concatenate([chunk, np.repeat(chunk[-1:], HOLD_STEPS, axis=0)])
        frames = [env.render()]
        for action in actions:
            env.step(action); frames.append(env.render())
        outputs.append((np.stack(frames).astype(np.uint8), actions.astype(np.float32)))
    return outputs


def render(archive):
    if RAW.exists():
        raise SystemExit(f"refusing to overwrite {RAW}")
    specs = sample_specs(); collected = {}
    with tempfile.TemporaryDirectory(prefix="panda_slot_data_") as temp:
        assets = extract_sim(archive, temp).parent
        for shape_index, shape in enumerate(SHAPES):
            env = PandaBlockPush(_shape_xml(assets, shape), width=WIDTH, height=HEIGHT, render=True)
            camera = env.model.camera("cam_fixed").id
            nominal_camera = env.model.cam_pos[camera].copy()
            nominal_light = env.model.light_diffuse.copy()
            try:
                selected = [row for row in specs if row["shape"] == shape_index]
                for done, spec in enumerate(selected, 1):
                    collected[spec["index"]] = _render_pair(
                        env, spec, nominal_camera, nominal_light)
                    print(f"render shape={shape_index} {done}/{len(selected)} pair={spec['index']}",
                          flush=True)
            finally:
                env.close()
    frames, actions, pair_index, split = [], [], [], []
    for index in range(PAIRS):
        for sequence_frames, sequence_actions in collected[index]:
            frames.append(sequence_frames); actions.append(sequence_actions)
            pair_index.append(index); split.append(0 if index < TRAIN_PAIRS else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, frames=np.stack(frames), actions=np.stack(actions),
                        pair_index=np.asarray(pair_index, np.int16),
                        split=np.asarray(split, np.uint8))
    print(f"wrote {RAW} ({RAW.stat().st_size / 2**20:.1f} MiB)")


def projection_matrix(device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    matrix = torch.randn(384, TOKEN_DIM, generator=generator, dtype=torch.float32)
    return (matrix / np.sqrt(TOKEN_DIM)).to(device)


def encode(checkpoint, batch_size=16, device="cpu"):
    if TOKENS.exists():
        raise SystemExit(f"refusing to overwrite {TOKENS}")
    data = np.load(RAW, allow_pickle=False)
    frames = data["frames"]
    model = load_world_model(checkpoint, device); encoder = model.encoder
    projection = projection_matrix(device); flat = frames.reshape(-1, *frames.shape[-3:])
    output = []
    for first in range(0, len(flat), batch_size):
        with torch.inference_mode():
            images = preprocess_frames(flat[first:first + batch_size, None], device)[:, 0]
            tokens = encoder.forward(images).float()
            tokens = torch.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
            output.append((tokens @ projection).half().cpu().numpy())
        print(f"encode {min(first + batch_size, len(flat))}/{len(flat)}", flush=True)
    descriptors = np.concatenate(output).reshape(*frames.shape[:2], 256, TOKEN_DIM)
    np.savez_compressed(TOKENS, descriptors=descriptors, actions=data["actions"],
                        pair_index=data["pair_index"], split=data["split"])
    print(f"wrote {TOKENS} ({TOKENS.stat().st_size / 2**20:.1f} MiB)")


def write_manifest():
    if DATA_MANIFEST.exists():
        raise SystemExit(f"refusing to overwrite {DATA_MANIFEST}")
    for path in (RAW, TOKENS):
        if not path.exists(): raise FileNotFoundError(path)
    value = {"protocol": PROTOCOL,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": {RAW.name: sha256(RAW), TOKENS.name: sha256(TOKENS)}}
    DATA_MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


def verify():
    value = json.loads(DATA_MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL:
        raise SystemExit("slot-data protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256(OUT / name) != expected:
            raise SystemExit(f"slot-data file changed: {name}")
    return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    encode_parser = sub.add_parser("encode")
    encode_parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    encode_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    encode_parser.add_argument("--batch-size", type=int, default=16)
    sub.add_parser("manifest"); sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "render": render(args.sim_archive)
    elif args.command == "encode": encode(args.checkpoint, args.batch_size, args.device)
    elif args.command == "manifest": write_manifest()
    else: print(verify()["files_sha256"])


if __name__ == "__main__":
    main()
