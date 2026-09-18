"""Bulk counterfactual data: new scene configurations, only the frames the models use.

PLAN_WORLDMODEL. Two changes from `jenga_gtf_data.py`:

* **New configurations.** The simulator's reset seed jitters block placement by +-2 mm and +-3
  degrees of yaw, which is the episode-to-episode variation in this dataset, and episodes 0-99 use
  seeds 0-99. Replaying a training episode's demo actions under seeds 100+ therefore gives
  genuinely new configurations with no policy and no contact with the holdout episodes.
* **Only the used frames.** W2 reads 3 history frames and the endings at hold 10 and 30, not all
  41, so this stores 5 latents per rollout instead of 41 (~8x smaller) and renders 2 frames per
  probe instead of 38 (much faster).

It also records the true block poses at both hold steps, so the same dataset serves the privileged
object-state probe (is localisation limited by the representation?) and the image models.

Simulation runs in worker processes; DINO encoding runs once in the parent on the GPU.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

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
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402

HOLD_STEPS = (10, 30)


def simulate(job):
    """One (episode, seed): every chunk start, K probes, 2 frames and 2 poses per probe."""
    episode_id, seed, lmdb, xml, snippets, scales, probes = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    actions_all = np.asarray(episode.actions, np.float32)
    starts = [s for s in chunk_starts(len(actions_all)) if s >= 2]
    sim = DirectJengaSim(xml)
    history_frames, history_proprio, start_pose = [], [], []
    probe_frames, probe_proprio, probe_pose, windows = [], [], [], []
    try:
        sim.reset(int(seed))
        frames, props = [sim.render()], [sim.proprio()]
        for step, action in enumerate(actions_all):
            if step in starts:
                snapshot = sim.snapshot()
                history_frames.append(np.stack(frames[-NUM_HIST:]))
                history_proprio.append(np.stack(props[-NUM_HIST:]))
                start_pose.append(all_block_pose(sim))
                chunk = actions_all[step:step + HORIZON]
                state_frames, state_props, state_pose, state_windows = [], [], [], []
                for probe in range(probes):
                    noise = snippets[probe % len(snippets)] * float(
                        scales[probe % len(scales)])
                    perturbed = chunk.copy(); perturbed[:, :3] -= noise
                    window = np.concatenate([actions_all[step - 2:step], perturbed,
                                             np.repeat(perturbed[-1:], HOLD, axis=0)])
                    sim.restore(snapshot)
                    for future in window[2:2 + HORIZON]:
                        sim.execute(future)
                    captured_frames, captured_props, captured_pose = [], [], []
                    for held in range(1, HOLD + 1):
                        sim.execute(window[-1])
                        if held in HOLD_STEPS:
                            captured_frames.append(sim.render())
                            captured_props.append(sim.proprio())
                            captured_pose.append(all_block_pose(sim))
                    state_frames.append(np.stack(captured_frames))
                    state_props.append(np.stack(captured_props))
                    state_pose.append(np.stack(captured_pose))
                    state_windows.append(window)
                sim.restore(snapshot)
                probe_frames.append(np.stack(state_frames))
                probe_proprio.append(np.stack(state_props))
                probe_pose.append(np.stack(state_pose))
                windows.append(np.stack(state_windows))
            sim.execute(action)
            frames.append(sim.render()); props.append(sim.proprio())
            frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
    finally:
        sim.close()
    return {"episode_id": episode_id, "seed": int(seed), "starts": np.asarray(starts),
            "history_frames": np.stack(history_frames),
            "history_proprio": np.stack(history_proprio),
            "start_pose": np.stack(start_pose), "probe_frames": np.stack(probe_frames),
            "probe_proprio": np.stack(probe_proprio), "probe_pose": np.stack(probe_pose),
            "actions": np.stack(windows)}


def encode(model, device, frames, proprio, batch_size=96):
    flat = frames.reshape(-1, *frames.shape[-3:])
    flat_p = proprio.reshape(-1, proprio.shape[-1])
    out = []
    for first in range(0, len(flat), batch_size):
        with torch.inference_mode():
            z = model.encode_obs({
                "visual": preprocess_frames(flat[first:first + batch_size, None], device),
                "proprio": normalise_proprio(flat_p[first:first + batch_size, None], device),
            })["visual"][:, 0]
        out.append(z.half().cpu().numpy())
    return np.concatenate(out).reshape(*frames.shape[:-3], -1, 384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"),
                    help="episodes IN this panel supply the demo action sequences")
    ap.add_argument("--out", default=str(ROOT / "results/jenga/bulk_data"))
    ap.add_argument("--seeds", type=int, default=10, help="new reset seeds per episode")
    ap.add_argument("--first-seed", type=int, default=100, help="0-99 are the bundle's episodes")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--snippet-seed", type=int, default=4321)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=0)
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
        if ep in set(train_episodes):
            continue
        design, errors = tracking_arrays([replay.episode(ep)])
        pools.append(errors - design @ coefficients)
    replay.close()
    snippets = sample_snippets(pools, count=64, seed=args.snippet_seed)
    scales = np.asarray(SCALES, np.float32)

    device = "cuda"
    model = load_world_model(args.checkpoint, device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [args.first_seed + i for i in range(args.seeds)]
    with tempfile.TemporaryDirectory(prefix="jenga_bulk_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        jobs = [(ep, seed, args.lmdb, xml, snippets, scales, args.probes)
                for seed in seeds for ep in train_episodes
                if not (out / f"ep{ep}_seed{seed}.npz").exists()]
        print(f"{len(jobs)} jobs ({len(train_episodes)} episodes x {len(seeds)} seeds)",
              flush=True)
        done = 0
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(simulate, jobs):
                history = encode(model, device, result["history_frames"],
                                 result["history_proprio"])
                probes = encode(model, device, result["probe_frames"], result["probe_proprio"])
                np.savez(out / f"ep{result['episode_id']}_seed{result['seed']}.npz",
                         history_latents=history, probe_latents=probes,
                         start_pose=result["start_pose"].astype(np.float32),
                         probe_pose=result["probe_pose"].astype(np.float32),
                         actions=result["actions"].astype(np.float32),
                         starts=result["starts"], episode_id=np.asarray(result["episode_id"]),
                         seed=np.asarray(result["seed"]))
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"  bulk {done}/{len(jobs)}", flush=True)
    meta = {"episodes": train_episodes, "seeds": seeds, "probes_per_state": args.probes,
            "hold_steps": list(HOLD_STEPS), "horizon": HORIZON, "hold": HOLD,
            "frames_stored": "3 history + 1 per hold step per probe",
            "physical_state": "all three block positions and rotation6 at each hold step",
            "seeds_note": "reset seeds 100+ are new configurations; 0-99 are the bundle episodes",
            "snippet_seed": args.snippet_seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
