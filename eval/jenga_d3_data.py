"""Generate task-label-free nested action-width supervision for D3.

Only TRAINING configurations from ``trace_data`` are used. At each contact-window state, several
realistic execution-error directions are swept along one scalar action line. The largest adjacent
generic trajectory response is selected without asking whether anything toppled or whether the
task succeeded. Five physical midpoint bisections then produce widths 1, 1/2, ..., 1/32.

Every saved level contains both physical endpoint trajectories through H=8 plus a common 30-step
hold. Smooth examples are retained: D3 matches the observed scale curve and never forces a plateau.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_cw_data import residual_windows, select_points  # noqa: E402

ACTION_HORIZON = 8
HOLD = 30
LEVELS = 6
COARSE_ALPHAS = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
STATE_GROUPS = (slice(0, 9), slice(9, 27), slice(27, 36), slice(36, 45))


def action_sequence(nominal, residual, alpha, hold=HOLD):
    """Perturb H=8 xyz targets and append the same nominal hold for every counterfactual."""
    nominal = np.asarray(nominal, np.float32)
    residual = np.asarray(residual, np.float32)
    if nominal.shape != (ACTION_HORIZON, 4) or residual.shape != (ACTION_HORIZON, 3):
        raise ValueError("nominal/residual must have shapes (8,4)/(8,3)")
    action = nominal.copy()
    action[:, :3] += float(alpha) * residual
    return np.concatenate([action, np.repeat(nominal[-1:], hold, axis=0)])


def generic_scales(trajectories):
    """One scale per anonymous rigid-body channel family, fitted within one coarse sweep."""
    values = np.asarray(trajectories, float)[..., :45]
    centred = values - values.mean(axis=0, keepdims=True)
    global_rms = max(float(np.sqrt(np.mean(centred ** 2))), 1e-8)
    scales = []
    for group in STATE_GROUPS:
        rms = float(np.sqrt(np.mean(centred[..., group] ** 2)))
        scales.append(max(rms, 1e-3 * global_rms, 1e-8))
    return np.asarray(scales)


def trajectory_distance(left, right, scales):
    """Equal-weight action/early/late discrepancy across generic state families."""
    difference = np.asarray(left, float)[..., :45] - np.asarray(right, float)[..., :45]
    phase_slices = (slice(0, ACTION_HORIZON), slice(ACTION_HORIZON, ACTION_HORIZON + 10),
                    slice(ACTION_HORIZON + 10, ACTION_HORIZON + HOLD))
    phase_values = []
    for phase in phase_slices:
        family_values = []
        for scale, group in zip(scales, STATE_GROUPS):
            family_values.append(np.sqrt(np.mean((difference[phase, group] / scale) ** 2)))
        phase_values.append(np.mean(family_values))
    return float(np.mean(phase_values))


def refine_line(left_alpha, right_alpha, left_trace, right_trace, simulate, scales,
                levels=LEVELS):
    """Bisect a physical response boundary and retain both endpoint trajectories at every width."""
    alphas = [[float(left_alpha), float(right_alpha)]]
    traces = [[np.asarray(left_trace), np.asarray(right_trace)]]
    templates = [np.asarray(left_trace), np.asarray(right_trace)]
    endpoints = [float(left_alpha), float(right_alpha)]
    endpoint_traces = [np.asarray(left_trace), np.asarray(right_trace)]
    midpoint_sides = []
    for _ in range(1, levels):
        midpoint = .5 * (endpoints[0] + endpoints[1])
        midpoint_trace = np.asarray(simulate(midpoint))
        distances = [trajectory_distance(midpoint_trace, template, scales)
                     for template in templates]
        side = int(np.argmin(distances))
        endpoints[side] = midpoint
        endpoint_traces[side] = midpoint_trace
        midpoint_sides.append(side)
        alphas.append(endpoints.copy())
        traces.append([endpoint_traces[0].copy(), endpoint_traces[1].copy()])
    return np.asarray(alphas, np.float32), np.asarray(traces, np.float32), midpoint_sides


def _simulate(sim, snapshot, warmstart, actions):
    from jenga_state_data import step_state
    sim.restore(snapshot); sim.data.qacc_warmstart[:] = warmstart
    states = []
    for action in actions:
        sim.execute(action); states.append(step_state(sim))
    return np.stack(states)


def simulate_file(job):
    path, points, xml, residuals, directions, seed = job
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim
    from jenga_state_data import step_state

    source = np.load(path, allow_pickle=False)
    episode_id, reset_seed = str(source["episode_id"]), int(source["seed"])
    starts, windows = source["starts"], source["actions"]
    replay = JengaReplay(DEFAULT_LMDB)
    episode_actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    rng = np.random.default_rng([seed, int(episode_id), reset_seed])
    sim = DirectJengaSim(xml)
    out = {name: [] for name in (
        "start", "actions", "traces", "alphas", "midpoint_sides", "coarse_score",
        "contact", "state_index", "probe_index", "contact_step")}
    try:
        needed = {int(starts[state_index]) for state_index, _, _, _ in points}
        clean = {}
        sim.reset(reset_seed)
        for step, action in enumerate(episode_actions):
            if step in needed:
                clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                if len(clean) == len(needed):
                    break
            sim.execute(action)

        for state_index, probe_index, contact_step, contact in points:
            snapshot, warm = clean[int(starts[state_index])]
            sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm
            for action in windows[state_index, probe_index, 2:2 + contact_step]:
                sim.execute(action)
            branch_snapshot = sim.snapshot()
            branch_warm = sim.data.qacc_warmstart.copy()
            start_state = step_state(sim)
            nominal = windows[state_index, probe_index,
                              2 + contact_step:2 + contact_step + ACTION_HORIZON].copy()
            if len(nominal) != ACTION_HORIZON:
                continue

            best = None
            for _ in range(directions):
                residual = residuals[int(rng.integers(len(residuals)))]
                coarse_actions = [action_sequence(nominal, residual, alpha)
                                  for alpha in COARSE_ALPHAS]
                coarse_traces = [_simulate(sim, branch_snapshot, branch_warm, actions)
                                 for actions in coarse_actions]
                scales = generic_scales(coarse_traces)
                for left in range(len(COARSE_ALPHAS) - 1):
                    score = trajectory_distance(coarse_traces[left], coarse_traces[left + 1],
                                                scales)
                    if best is None or score > best[0]:
                        best = (score, residual.copy(), left, coarse_traces[left].copy(),
                                coarse_traces[left + 1].copy(), scales.copy())

            score, residual, left, left_trace, right_trace, scales = best
            def simulate(alpha):
                return _simulate(sim, branch_snapshot, branch_warm,
                                 action_sequence(nominal, residual, alpha))

            alphas, traces, sides = refine_line(
                COARSE_ALPHAS[left], COARSE_ALPHAS[left + 1], left_trace, right_trace,
                simulate, scales)
            nested_actions = np.stack([[action_sequence(nominal, residual, alpha)
                                        for alpha in pair] for pair in alphas])
            out["start"].append(start_state); out["actions"].append(nested_actions)
            out["traces"].append(traces); out["alphas"].append(alphas)
            out["midpoint_sides"].append(sides); out["coarse_score"].append(score)
            out["contact"].append(contact); out["state_index"].append(state_index)
            out["probe_index"].append(probe_index); out["contact_step"].append(contact_step)
    finally:
        sim.close()
    floats = {"start", "actions", "traces", "alphas", "coarse_score"}
    return path, {name: np.asarray(values, np.float32 if name in floats else None)
                  for name, values in out.items()}


def limited_points(points, maximum, seed):
    """Deterministically cap total groups while retaining the original per-file structure."""
    flat = [(path, point) for path in sorted(points) for point in points[path]]
    if maximum and len(flat) > maximum:
        rng = np.random.default_rng(seed)
        chosen = set(map(int, rng.choice(len(flat), maximum, replace=False)))
        flat = [row for index, row in enumerate(flat) if index in chosen]
    selected = {path: [] for path in points}
    for path, point in flat:
        selected[path].append(point)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--output", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--quiet-fraction", type=float, default=0.15)
    parser.add_argument("--residuals", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="keep at one on desktops; each worker owns a MuJoCo instance")
    args = parser.parse_args()
    if args.directions < 1 or args.max_points < 1:
        parser.error("directions and max-points must be positive")

    from jenga_short_held_tails import extract_sim
    points, contacts, quiet = select_points(args.trace, args.quiet_fraction, args.seed)
    points = limited_points(points, args.max_points, args.seed)
    selected = sum(map(len, points.values()))
    residuals = residual_windows(args.residuals, ACTION_HORIZON, args.seed + 1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jenga_d3_") as temp:
        xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temp))
        jobs = [(path, selected_points, xml, residuals, args.directions, args.seed)
                for path, selected_points in points.items() if selected_points]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            total = 0
            for done, (path, arrays) in enumerate(pool.map(simulate_file, jobs), 1):
                target = output / Path(path).name
                np.savez_compressed(target, **arrays); total += len(arrays["start"])
                print(f"D3 files {done}/{len(jobs)}, groups {total}/{selected}", flush=True)
    meta = {
        "source": str(Path(args.trace).resolve()), "task_labels": False,
        "selection": "largest adjacent generic trajectory response along realistic action lines",
        "action_horizon": ACTION_HORIZON, "hold": HOLD, "levels": LEVELS,
        "widths": [2.0 ** -level for level in range(LEVELS)],
        "coarse_alphas": COARSE_ALPHAS.tolist(), "directions": args.directions,
        "max_points": args.max_points, "selected_points": selected,
        "available_contact_points": contacts, "available_quiet_points": quiet,
        "quiet_fraction": args.quiet_fraction, "residuals": args.residuals,
        "seed": args.seed, "workers": args.workers,
        "layout": "start (G,61); actions (G,L,2,38,4); traces (G,L,2,38,61)",
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
