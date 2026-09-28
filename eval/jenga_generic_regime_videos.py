"""Render paired final-bracket rollouts from the generic physical-regime audit."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_generic_regime_audit import reconstruct_endpoints  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD  # noqa: E402

CASES = (
    ("quiet_contact_strong", "17", 122, 2),
    ("quiet_contact_moderate", "44", 90, 0),
    ("quiet_motion_transient_a", "26", 50, 1),
    ("quiet_motion_transient_b", "96", 66, 0),
    ("fork_reference", "17", 106, 1),
    ("quiet_control_reference", "80", 66, 0),
)
FPS = 6


def _phase(step):
    if step == 0:
        return "start"
    if step <= HORIZON:
        return f"action {step}/{HORIZON}"
    return f"hold {step - HORIZON}/{HOLD}"


def _panel(frame, side, step, contacts, case, metrics):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    canvas = np.full((image.shape[0] + 92, image.shape[1], 3), 245, np.uint8)
    canvas[92:] = image
    cv2.putText(canvas, f"{case}  |  side {side}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .47, (25, 25, 25), 1)
    cv2.putText(canvas, f"{_phase(step)}   active contact edges: {contacts}", (8, 43),
                cv2.FONT_HERSHEY_SIMPLEX, .44, (25, 25, 25), 1)
    cv2.putText(canvas, f"pair: persistent contact={str(metrics['persistent_contact_difference'])}",
                (8, 65), cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1)
    cv2.putText(canvas, f"final pose gap={metrics['pose_gap_final']:.3f}", (8, 84),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1)
    return canvas


def _sequence(sim, snapshot, chunk, noise):
    sim.restore(snapshot)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    frames = [sim.render()]
    contacts = [int(np.sum(sim.contact_signature()))]
    for action in actions:
        sim.execute(action); frames.append(sim.render())
        contacts.append(int(np.sum(sim.contact_signature())))
    for _ in range(HOLD):
        sim.execute(chunk[-1]); frames.append(sim.render())
        contacts.append(int(np.sum(sim.contact_signature())))
    return frames, contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--audit", default=str(ROOT / "results/jenga/generic_regime_audit_dev.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--output-dir", default=str(
        ROOT / "results/jenga/generic_regime_videos"))
    args = ap.parse_args()
    rows = json.loads(Path(args.audit).read_text())["rows"]
    index = {(r["episode_id"], r["chunk_start"]): r for r in rows}
    snippets = np.load(args.cache, allow_pickle=False)["snippets"]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = []
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_regime_videos_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for case, episode_id, start, pair_index in CASES:
                    row = index[(episode_id, start)]
                    pair_noises = reconstruct_endpoints(snippets, row["pair_details"])[pair_index]
                    episode = replay.episode(episode_id)
                    sim.reset(int(episode_id))
                    for action in episode.actions[:start]:
                        sim.execute(action)
                    snapshot = sim.snapshot()
                    chunk = episode.actions[start:start + HORIZON]
                    sequences = [_sequence(sim, snapshot, chunk, noise) for noise in pair_noises]
                    metrics = row["pair_metrics"][pair_index]
                    name = f"{case}_ep{episode_id}_chunk{start}_pair{pair_index}.mp4"
                    path = output / name
                    height, width = sequences[0][0][0].shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                             (2 * width, height + 92))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 writer unavailable")
                    try:
                        for step in range(HORIZON + HOLD + 1):
                            panels = [_panel(frames[step], side, step, contacts[step], case, metrics)
                                      for side, (frames, contacts) in
                                      zip(("A", "B"), sequences)]
                            joined = np.concatenate(panels, axis=1)
                            repeats = FPS if step in (0, HORIZON, HORIZON + HOLD) else 1
                            for _ in range(repeats):
                                writer.write(joined)
                    finally:
                        writer.release()
                    manifest.append({"video": name, "case": case, "episode_id": episode_id,
                                     "chunk_start": start, "pair_index": pair_index,
                                     "cohort": row["cohort"], "class": row["class"],
                                     "metrics": metrics})
                    print(f"wrote {name}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
