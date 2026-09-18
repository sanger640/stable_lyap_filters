"""Full privileged state for the localisation probe: blocks, gripper, velocities and contacts.

The first version of the probe fed only the three blocks' poses at the chunk start, in world
coordinates, while the actions were expressed relative to the gripper. That model could not compute
the gripper-to-block geometry that decides whether a finger catches a block, had no velocities (so
a rocking block's future was not determined by its input at all) and no contact flags. This
records the state that actually determines the outcome:

  * block positions and rotation6, in world coordinates AND relative to the gripper
  * gripper position and gripper open/closed
  * block linear and angular velocities
  * the contact signature (which bodies touch)

No rendering and no encoding, so it runs in minutes. Same episodes, seeds, snippets and probes as
`jenga_bulk_data.py`, so the two datasets line up state for state.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from action_uncertainty import tracking_arrays  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_bulk_data import HOLD_STEPS  # noqa: E402
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, chunk_starts,  # noqa: E402
                                    extract_sim)
from jenga_stage0_noise_oracle import HOLD, SCALES, sample_snippets  # noqa: E402


def block_velocities(sim):
    """Linear and angular velocity of each tracked block, from its free joint."""
    out = []
    for body in sim.tracked_block_ids:
        joint = sim.model.body_jntadr[body]
        address = sim.model.jnt_dofadr[joint]
        out.append(sim.data.qvel[address:address + 6].copy())
    return np.concatenate(out).astype(np.float32)


def full_state(sim):
    """Everything that determines the outcome, in one vector."""
    pose = all_block_pose(sim)                      # 9 positions + 18 rotation columns
    gripper = sim.proprio()                         # end-effector xyz + open/closed
    positions = pose[:9].reshape(3, 3)
    relative = (positions - gripper[:3][None]).reshape(-1)   # blocks relative to the gripper
    return np.concatenate([pose, relative, gripper, block_velocities(sim),
                           sim.contact_signature().astype(np.float32)]).astype(np.float32)


def step_state(sim):
    """Per-step state for a learned integrator: block pose, velocities, contacts, gripper."""
    return np.concatenate([all_block_pose(sim), block_velocities(sim),
                           sim.contact_signature().astype(np.float32),
                           sim.proprio()]).astype(np.float32)


def simulate(job):
    episode_id, seed, lmdb, xml, snippets, scales, probes, per_step = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    actions_all = np.asarray(episode.actions, np.float32)
    starts = [s for s in chunk_starts(len(actions_all)) if s >= 2]
    sim = DirectJengaSim(xml)
    start_state, ending_pose, windows, traces = [], [], [], []
    try:
        sim.reset(int(seed))
        for step, action in enumerate(actions_all):
            if step in starts:
                snapshot = sim.snapshot()
                start_state.append(full_state(sim))
                chunk = actions_all[step:step + HORIZON]
                state_endings, state_windows, state_traces = [], [], []
                for probe in range(probes):
                    noise = snippets[probe % len(snippets)] * float(scales[probe % len(scales)])
                    perturbed = chunk.copy(); perturbed[:, :3] -= noise
                    window = np.concatenate([actions_all[step - 2:step], perturbed,
                                             np.repeat(perturbed[-1:], HOLD, axis=0)])
                    sim.restore(snapshot)
                    chunk_trace = []
                    for future in window[2:2 + HORIZON]:
                        sim.execute(future)
                        if per_step:
                            chunk_trace.append(step_state(sim))
                    captured, trajectory = [], []
                    if per_step:
                        trajectory.extend(chunk_trace)
                    for held in range(1, HOLD + 1):
                        sim.execute(window[-1])
                        if per_step:
                            trajectory.append(step_state(sim))
                        if held in HOLD_STEPS:
                            captured.append(all_block_pose(sim))
                    state_endings.append(np.stack(captured))
                    if per_step:
                        state_traces.append(np.stack(trajectory))
                    state_windows.append(window)
                sim.restore(snapshot)
                ending_pose.append(np.stack(state_endings))
                windows.append(np.stack(state_windows))
                if per_step:
                    traces.append(np.stack(state_traces))
            sim.execute(action)
    finally:
        sim.close()
    out = {"episode_id": episode_id, "seed": int(seed), "starts": np.asarray(starts),
           "start_state": np.stack(start_state).astype(np.float32),
           "probe_pose": np.stack(ending_pose).astype(np.float32),
           "actions": np.stack(windows).astype(np.float32)}
    if per_step:
        out["traces"] = np.stack(traces).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--out", default=str(ROOT / "results/jenga/state_data"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--snippet-seed", type=int, default=4321)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-step", action="store_true",
                    help="record the state after every action: training data for an integrator")
    args = ap.parse_args()

    train_episodes = sorted({r["episode_id"] for r in
                             json.loads(Path(args.panel).read_text())["rows"]}, key=int)
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
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [args.first_seed + i for i in range(args.seeds)]
    with tempfile.TemporaryDirectory(prefix="jenga_state_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        jobs = [(ep, seed, args.lmdb, xml, snippets, scales, args.probes, args.per_step)
                for seed in seeds for ep in train_episodes]
        print(f"{len(jobs)} jobs", flush=True)
        done = 0
        with ProcessPoolExecutor(args.workers) as pool:
            for result in pool.map(simulate, jobs):
                np.savez(out / f"ep{result['episode_id']}_seed{result['seed']}.npz", **result)
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"  state {done}/{len(jobs)}", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"episodes": train_episodes, "seeds": seeds, "probes_per_state": args.probes,
         "state_layout": "27 block pose | 9 blocks relative to gripper | 4 gripper | "
                         "18 block velocities | 12 contact flags",
         "hold_steps": list(HOLD_STEPS), "snippet_seed": args.snippet_seed}, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
