"""Build rollout fine-tuning data for the Jenga predictor: perturbed chunks WITH settle tails.

The recorded demos contain no held poses (median per-step EE displacement 5.3 mm, longest run of
sub-millimetre steps = 1), so the 30-step hold the monitor relies on is out of distribution for
the shipped checkpoint. This regenerates it in the simulator.

Only the 43 development-panel episodes are used, so the holdout episodes stay unseen. Each item is
one chunk: three real history frames, then H=8 perturbed actions and HOLD held steps, every frame
rendered from the world model's camera and DINOv2-encoded (the encoder is frozen, so latents are
precomputed once). Perturbations are tracking-residual snippets drawn with a DIFFERENT seed from
the 64 used in evaluation, at a scale sampled from 0.5/1/2.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from action_uncertainty import tracking_arrays  # noqa: E402
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           NUM_HIST, load_world_model, normalise_proprio, preprocess_frames)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import (DirectJengaSim, HORIZON, chunk_starts,  # noqa: E402
                                    extract_sim)
from jenga_stage0_noise_oracle import HOLD, SCALES, sample_snippets  # noqa: E402


def encode_frames(model, device, frames, proprio, batch_size=64):
    out = []
    for first in range(0, len(frames), batch_size):
        with torch.inference_mode():
            z = model.encode_obs({
                "visual": preprocess_frames(frames[first:first + batch_size, None], device),
                "proprio": normalise_proprio(proprio[first:first + batch_size, None], device),
            })["visual"][:, 0]
        out.append(z.half().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"),
                    help="episodes IN this panel are the TRAINING episodes")
    ap.add_argument("--out", default=str(ROOT / "results/jenga/gtf_data"))
    ap.add_argument("--probes-per-chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes", type=int, default=0, help="first N training episodes only")
    ap.add_argument("--chunk-stride", type=int, default=1,
                    help="keep every Nth chunk start (use with a large --probes-per-chunk)")
    args = ap.parse_args()

    train_episodes = sorted({r["episode_id"] for r in
                             json.loads(Path(args.panel).read_text())["rows"]}, key=int)
    if args.episodes:
        train_episodes = train_episodes[:args.episodes]
    replay = JengaReplay(args.lmdb)
    coefficients = np.asarray(json.loads(
        (ROOT / "results/jenga/tracking_uncertainty.json").read_text())["lag_coefficients"])
    pools = []
    for ep in replay.episode_ids:
        if ep in set(train_episodes):  # residuals from the OTHER episodes, as in evaluation
            continue
        design, errors = tracking_arrays([replay.episode(ep)])
        pools.append(errors - design @ coefficients)
    snippets = sample_snippets(pools, count=256, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda"
    model = load_world_model(args.checkpoint, device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import tempfile
    written = 0
    with tempfile.TemporaryDirectory(prefix="jenga_gtf_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(train_episodes):
                target = out / f"ep{episode_id}.npz"
                if target.exists():
                    written += 1
                    continue
                episode = replay.episode(episode_id)
                starts = set(chunk_starts(len(episode.actions)))
                sim.reset(int(episode_id))
                frames, props = [sim.render()], [sim.proprio()]
                latents, actions, proprios, state_ids = [], [], [], []
                kept = 0
                for step, action in enumerate(episode.actions):
                    if step in starts and step >= 2:
                        kept += 1
                        if (kept - 1) % args.chunk_stride:
                            sim.execute(action)
                            frames.append(sim.render()); props.append(sim.proprio())
                            frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
                            continue
                        snapshot = sim.snapshot()
                        history_frames = np.stack(frames[-NUM_HIST:])
                        history_proprio = np.stack(props[-NUM_HIST:])
                        chunk = np.asarray(episode.actions[step:step + HORIZON], np.float32)
                        for _ in range(args.probes_per_chunk):
                            noise = snippets[rng.integers(len(snippets))] * float(
                                SCALES[rng.integers(len(SCALES))])
                            probe = chunk.copy(); probe[:, :3] -= noise
                            window = np.concatenate([
                                np.asarray(episode.actions[step - 2:step], np.float32), probe,
                                np.repeat(probe[-1:], HOLD, axis=0)])
                            sim.restore(snapshot)
                            seq_frames, seq_props = [], []
                            for future in window[2:]:
                                sim.execute(future)
                                seq_frames.append(sim.render()); seq_props.append(sim.proprio())
                            sim.restore(snapshot)
                            all_frames = np.concatenate([history_frames, np.stack(seq_frames)])
                            all_props = np.concatenate([history_proprio, np.stack(seq_props)])
                            latents.append(encode_frames(model, device, all_frames, all_props))
                            actions.append(window); proprios.append(all_props)
                            state_ids.append(f"{episode_id}:{step}")
                    sim.execute(action)
                    frames.append(sim.render()); props.append(sim.proprio())
                    frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
                if latents:
                    np.savez(target, latents=np.stack(latents),
                             actions=np.stack(actions).astype(np.float32),
                             proprio=np.stack(proprios).astype(np.float32),
                             state_ids=np.asarray(state_ids),
                             episode_id=np.asarray(episode_id))
                    written += 1
                print(f"  gtf data {number + 1}/{len(train_episodes)} ep{episode_id} "
                      f"items={len(latents)}", flush=True)
        finally:
            sim.close()
            replay.close()
    meta = {"training_episodes": train_episodes, "probes_per_chunk": args.probes_per_chunk,
            "chunk_stride": args.chunk_stride,
            "horizon": HORIZON, "hold": HOLD, "history": NUM_HIST, "seed": args.seed,
            "snippet_pool": "residuals from the NON-training episodes, seed differs from the "
                            "64 evaluation snippets", "files": written,
            "latent_layout": "index 0..2 history frames, then one frame after each future action"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
