"""Contact-window counterfactual branches for training (PLAN_NEXT.md Phase 3, revised step 7).

`blind_fork_analysis.md` located the dynamics failure inside the contact window: while the gripper
is pushing a block, the model under-delivers the rotation, and the only same-state /
different-action pairs in the training data sit at the chunk start, before that window. This
generates branches FROM TRUE STATES INSIDE THE WINDOW:

  branch points  training (state, probe) paths from `results/jenga/trace_data` whose gripper
                 touches a neighbour block during control steps 3-10. The branch state is taken
                 just before that contact (step k in 3..10). A share of window states WITHOUT
                 contact is added, so the data also contains interventions that change nothing.
  branches       from the identical branch state, H control steps of the path's own next actions
                 (nominal), plus M antithetic pairs u +/- d, where d is a realistic execution-error
                 residual window x a scale in {0.5, 1, 2}. 1 + 2M branches per point.
  recorded       the full 61-dim state after every control step: pose, 6D orientation, linear and
                 angular velocity, contacts, gripper.

Only TRAINING configurations are used (the 43 panel episodes, reset seeds 100-109); none overlaps
the benchmark's dev or test episodes. Perturbations are a fresh draw of residual windows from the
same execution-error model as the training data, not the benchmark's 64 probe snippets.

Warm-start-safe: every chunk start is reached by an UNINTERRUPTED replay; the path to the branch
state is executed from that clean snapshot; every branch then restores the branch snapshot together
with the solver warm start, so all branches of a point start from an identical simulator state.
Contact flags select branch points only; they are not used by any loss.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

NEIGHBOUR_ROBOT = (45 + 10, 45 + 11)        # left-robot, right-robot contact flags
WINDOW = (3, 10)                            # control steps the branch state is drawn from
SCALES = (0.5, 1.0, 2.0)


def select_points(trace_dir, quiet_fraction, seed):
    """-> {file: [(state_index, probe, k, contact)]}, chosen from the recorded training paths."""
    rng = np.random.default_rng(seed)
    points, n_contact, n_quiet = {}, 0, 0
    for path in sorted(glob.glob(str(Path(trace_dir) / "ep*_seed*.npz"))):
        traces = np.load(path, allow_pickle=False)["traces"]            # (S, K, 38, 61)
        touch = (traces[..., NEIGHBOUR_ROBOT[0]] > 0.5) | (traces[..., NEIGHBOUR_ROBOT[1]] > 0.5)
        chosen = []
        for s in range(traces.shape[0]):
            for p in range(traces.shape[1]):
                # traces[..., t] is the state after t + 1 actions
                steps = [t + 1 for t in range(WINDOW[0] - 1, WINDOW[1]) if touch[s, p, t]]
                if steps:
                    k = int(np.clip(steps[0] - 1, *WINDOW))            # just before contact
                    chosen.append((s, p, k, True)); n_contact += 1
                elif rng.random() < quiet_fraction:
                    chosen.append((s, p, int(rng.integers(WINDOW[0], WINDOW[1] + 1)), False))
                    n_quiet += 1
        points[path] = chosen
    return points, n_contact, n_quiet


def residual_windows(count, horizon, seed):
    """Fresh execution-error residual windows from the non-training episodes' tracking residuals."""
    from action_uncertainty import tracking_arrays
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_stage0_noise_oracle import sample_snippets
    train = set(json.loads((ROOT / "results/jenga/trace_data/meta.json").read_text())["episodes"])
    coefficients = np.asarray(json.loads(
        (ROOT / "results/jenga/tracking_uncertainty.json").read_text())["lag_coefficients"])
    replay = JengaReplay(DEFAULT_LMDB)
    pools = []
    for ep in replay.episode_ids:
        if ep in train:
            continue
        design, errors = tracking_arrays([replay.episode(ep)])
        pools.append(errors - design @ coefficients)
    replay.close()
    return sample_snippets(pools, count=count, horizon=horizon, seed=seed)


def simulate_file(job):
    path, points, xml, residuals, horizon, pairs, seed = job
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim
    from jenga_state_data import step_state
    data = np.load(path, allow_pickle=False)
    episode_id, reset_seed = str(data["episode_id"]), int(data["seed"])
    starts, windows = data["starts"], data["actions"]                   # windows (S, K, 40, 4)
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    rng = np.random.default_rng([seed, int(episode_id), reset_seed])
    sim = DirectJengaSim(xml)
    out = {k: [] for k in ("start", "actions", "traces", "contact", "k", "scale")}
    try:
        # Pass 1: uninterrupted replay, clean snapshot at every needed chunk start.
        needed = {int(starts[s]) for s, _, _, _ in points}
        clean = {}
        sim.reset(reset_seed)
        for step, action in enumerate(actions):
            if step in needed:
                clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                if len(clean) == len(needed):
                    break
            sim.execute(action)
        # Pass 2: path to the branch state, then every branch from that identical state.
        for s, p, k, contact in points:
            snapshot, warm = clean[int(starts[s])]
            sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm
            for a in windows[s, p, 2:2 + k]:
                sim.execute(a)
            branch_snapshot, branch_warm = sim.snapshot(), sim.data.qacc_warmstart.copy()
            start = step_state(sim)
            nominal = windows[s, p, 2 + k:2 + k + horizon].copy()
            branch_actions, scales = [nominal], []
            for _ in range(pairs):
                r = residuals[int(rng.integers(len(residuals)))][:horizon]
                scale = float(rng.choice(SCALES)); scales.append(scale)
                for sign in (-1.0, 1.0):
                    a = nominal.copy(); a[:, :3] += sign * scale * r
                    branch_actions.append(a)
            traces = []
            for a_seq in branch_actions:
                sim.restore(branch_snapshot); sim.data.qacc_warmstart[:] = branch_warm
                trace = []
                for a in a_seq:
                    sim.execute(a)
                    trace.append(step_state(sim))
                traces.append(np.stack(trace))
            out["start"].append(start); out["actions"].append(np.stack(branch_actions))
            out["traces"].append(np.stack(traces)); out["contact"].append(contact)
            out["k"].append(k); out["scale"].append(scales)
    finally:
        sim.close()
    floats = ("start", "actions", "traces", "scale")
    return path, {k: np.asarray(v, np.float32 if k in floats else None) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    ap.add_argument("--out", default=str(ROOT / "results/jenga/cw_data"))
    ap.add_argument("--horizon", type=int, default=6, help="control steps per branch (3-6)")
    ap.add_argument("--pairs", type=int, default=2, help="antithetic +/- pairs per branch point")
    ap.add_argument("--quiet-fraction", type=float, default=0.15,
                    help="probability of adding a no-contact window state per non-contact path")
    ap.add_argument("--residuals", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from jenga_short_held_tails import extract_sim
    points, n_contact, n_quiet = select_points(args.trace, args.quiet_fraction, args.seed)
    residuals = residual_windows(args.residuals, args.horizon, args.seed + 1)
    print(f"{n_contact} contact + {n_quiet} no-contact branch points over {len(points)} files, "
          f"{1 + 2 * args.pairs} branches x {args.horizon} steps each", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stats = {"contact_points": n_contact, "quiet_points": n_quiet}
    with tempfile.TemporaryDirectory(prefix="jenga_cw_") as temp:
        xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temp))
        jobs = [(path, pts, xml, residuals, args.horizon, args.pairs, args.seed)
                for path, pts in points.items() if pts]
        with ProcessPoolExecutor(args.workers) as pool:
            for done, (path, result) in enumerate(pool.map(simulate_file, jobs), 1):
                np.savez_compressed(out / Path(path).name, **result)
                if done % 25 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} files", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"source": str(Path(args.trace).relative_to(ROOT)), "horizon": args.horizon,
         "pairs": args.pairs, "branches_per_point": 1 + 2 * args.pairs, "window": list(WINDOW),
         "quiet_fraction": args.quiet_fraction, "scales": list(SCALES),
         "residual_windows": args.residuals, "seed": args.seed, **stats,
         "layout": "start (G,61); actions (G,B,H,4) branch 0 nominal, then -d/+d pairs; "
                   "traces (G,B,H,61) state after each control step"}, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
