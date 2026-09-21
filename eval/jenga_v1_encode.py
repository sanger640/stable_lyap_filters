"""Render and encode the chunk-start frames at a chosen resolution, for the V1 resolution sweep.

The bundled world model always squeezes its input to 196x196 (14x14 patches of 14 px), so the
geometry a probe can read is capped by that grid whatever the camera does. This calls the frozen
DINOv2 encoder DIRECTLY, bypassing that resize, and renders the scene at a matching camera
resolution -- upsampling a 240x320 render to 392 px would add pixels but no information.

Only the encoder is used. The DINO-WM predictor and decoder are not involved: V1 treats DINOv2 as
a frozen task-agnostic feature extractor and the dynamics stay the privileged-state model.

Writes one npz per episode/seed with the patch tokens and the true 61-dim state, so
`jenga_v1_probe.py` and `jenga_v1_gate.py` can consume either resolution unchanged.
"""
import argparse
from pathlib import Path
import os
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           NUM_HIST, load_world_model)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim  # noqa: E402
from jenga_state_data import step_state  # noqa: E402


def preprocess(frames, device, pixels):
    """uint8 (N,H,W,3) -> the encoder's [-1,1] square input at `pixels`, no model-side resize."""
    from torchvision.transforms import functional as TF
    x = torch.as_tensor(np.asarray(frames), device=device).permute(0, 3, 1, 2).float() / 255.0
    x = TF.resize(x, [pixels, pixels], antialias=True)
    return (x - 0.5) / 0.5


def encode(encoder, frames, device, pixels, batch=16):
    out = []
    for first in range(0, len(frames), batch):
        with torch.inference_mode():
            z = encoder.forward(preprocess(frames[first:first + batch], device, pixels))
        out.append(z.half().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--pixels", type=int, default=196, help="encoder input side, a multiple of 14")
    ap.add_argument("--render-height", type=int, default=240)
    ap.add_argument("--render-width", type=int, default=320)
    ap.add_argument("--frames", type=int, default=1, help="frames ending at the chunk start")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=100)
    args = ap.parse_args()

    import json
    episodes = sorted({r["episode_id"] for r in
                       json.loads(Path(args.panel).read_text())["rows"]}, key=int)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_world_model(args.checkpoint, device).encoder
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    replay = JengaReplay(args.lmdb)

    with tempfile.TemporaryDirectory(prefix="jenga_enc_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        done = 0
        for seed in [args.first_seed + i for i in range(args.seeds)]:
            for episode_id in episodes:
                target = out / f"ep{episode_id}_seed{seed}.npz"
                if target.exists():
                    done += 1
                    continue
                episode = replay.episode(episode_id)
                actions = np.asarray(episode.actions, np.float32)
                starts = [s for s in chunk_starts(len(actions)) if s >= 2]
                sim = DirectJengaSim(xml)
                sim.renderer.close()
                import mujoco
                sim.renderer = mujoco.Renderer(sim.model, height=args.render_height,
                                               width=args.render_width)
                try:
                    sim.reset(int(seed))
                    history, latents, states = [sim.render()], [], []
                    for step, action in enumerate(actions):
                        if step in starts:
                            window = history[-args.frames:]
                            while len(window) < args.frames:
                                window = [window[0]] + window
                            latents.append(encode(encoder, np.stack(window), device, args.pixels))
                            states.append(step_state(sim))
                        sim.execute(action)
                        history.append(sim.render())
                        history = history[-max(NUM_HIST, args.frames):]
                finally:
                    sim.close()
                np.savez(target, latents=np.stack(latents).astype(np.float16),
                         state=np.stack(states).astype(np.float32),
                         starts=np.asarray(starts), episode_id=episode_id, seed=int(seed),
                         pixels=args.pixels)
                done += 1
                if done % 25 == 0:
                    print(f"  encoded {done}/{args.seeds * len(episodes)}", flush=True)
    replay.close()
    print("done")


if __name__ == "__main__":
    main()
