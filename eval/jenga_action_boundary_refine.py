"""Adaptive action-boundary refinement on ground-truth DEV trajectories.

This is a universal structural test: repeatedly bisect nearby actions whose trajectories occupy
opposite response groups, then ask whether the trajectory gap decays to zero or approaches a
nonzero plateau. Jenga classes are attached only after the alarm for evaluation. TEST is not read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,  # noqa: E402
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from jenga_action_branch import action_errors, pose_features, summarise  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, SPLITS, verify  # noqa: E402
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT, rollout  # noqa: E402

PAIRS = 3
REFINEMENTS = 5


def _projector(reference, dimensions=6):
    flat = np.asarray(reference, float).reshape(len(reference), -1)
    mean = flat.mean(0)
    centred = flat - mean
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:min(dimensions, len(axes))]
    coordinates = centred @ axes.T
    keep = coordinates.std(0) > 1e-10
    axes = axes[keep]
    if not len(axes):
        axes = np.zeros((1, flat.shape[1]))
    coordinates = centred @ axes.T
    scale = max(float(np.sqrt(np.mean(coordinates ** 2))), 1e-10)

    def project(values):
        values = np.asarray(values, float).reshape(len(values), -1)
        return ((values - mean) @ axes.T) / scale
    return project


def _refine_state(sim, snapshot, chunk, snippets, original_pose, labels, pairs,
                  start_pose, start_tilt):
    features = pose_features(original_pose)
    early_stop = RECORD_AT.index(10) + 1
    early_project = _projector(features[:, :early_stop])
    full_project = _projector(features)
    original_early = early_project(features[:, :early_stop])
    original_full = full_project(features)
    centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
    early_gaps, full_gaps, pair_rows = [], [], []
    for left, right in pairs:
        endpoint_noise = [snippets[left].copy(), snippets[right].copy()]
        endpoint_features = [features[left].copy(), features[right].copy()]
        pair_early = [float(np.linalg.norm(original_early[left] - original_early[right]))]
        pair_full = [float(np.linalg.norm(original_full[left] - original_full[right]))]
        midpoint_sides = []
        for _ in range(REFINEMENTS):
            midpoint_noise = .5 * (endpoint_noise[0] + endpoint_noise[1])
            midpoint_pose, _, _ = rollout(sim, snapshot, chunk, midpoint_noise,
                                           start_pose, start_tilt)
            midpoint = pose_features(midpoint_pose)
            midpoint_full = full_project(midpoint[None])[0]
            side = int(np.argmin(np.linalg.norm(centroids - midpoint_full[None], axis=1)))
            endpoint_noise[side] = midpoint_noise
            endpoint_features[side] = midpoint
            endpoint_early = early_project(np.stack(
                [value[:early_stop] for value in endpoint_features]))
            endpoint_full = full_project(np.stack(endpoint_features))
            pair_early.append(float(np.linalg.norm(endpoint_early[0] - endpoint_early[1])))
            pair_full.append(float(np.linalg.norm(endpoint_full[0] - endpoint_full[1])))
            midpoint_sides.append(side)
        early_gaps.append(pair_early); full_gaps.append(pair_full)
        pair_rows.append({"probe_indices": [int(left), int(right)],
                          "midpoint_sides": midpoint_sides,
                          "early_gaps": pair_early, "full_gaps": pair_full})
    evidence = boundary_refinement_alarm(early_gaps, full_gaps)
    return result_dict(evidence), pair_rows


def _run_episode(job):
    episode_id, targets, lmdb, xml, snippets = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id))
        for step, action in enumerate(episode.actions):
            if step in targets:
                target = targets[step]
                snapshot = sim.snapshot()
                start_pose = all_block_pose(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                result, pair_rows = _refine_state(
                    sim, snapshot, episode.actions[step:step + HORIZON], snippets,
                    target["pose"], target["labels"], target["pairs"], start_pose, start_tilt)
                out[step] = {**result, "pair_details": pair_rows}
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def prepare(limit_alarmed=0):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    spec = SPLITS["dev"]
    stage0 = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
    cache_index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0)}
    cache = np.load(ROOT / spec["cache"], allow_pickle=False)
    scale_index = EVAL_CONFIG["scales"].index(1.0)
    rows, wanted, alarmed = [], {}, 0
    for i, episode_id in enumerate(bench["dev_episode"]):
        chunk = int(bench["dev_chunk"][i])
        cache_row = cache_index[(str(episode_id), chunk)]
        pose = cache["pose"][cache_row, scale_index]
        windows = bench["dev_windows"][i, scale_index]
        features = pose_features(pose)
        early_stop = RECORD_AT.index(10) + 1
        initial = smooth_vs_branch_alarm(
            action_errors(windows), features[:, :early_stop], features)
        row = {"split": "dev", "episode_id": str(episode_id), "chunk_start": chunk,
               "scale": "1.0", "class": str(bench["dev_classes"][i, scale_index]),
               "topple_count": int(bench["dev_topples"][i, scale_index]),
               "initial_alarm": bool(initial.alarm), "alarm": False}
        rows.append(row)
        if not initial.alarm or (limit_alarmed and alarmed >= limit_alarmed):
            continue
        x, labels = action_branch_partition(
            action_errors(windows), features[:, :early_stop], features)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        wanted.setdefault(str(episode_id), {})[chunk] = {
            "pose": pose, "labels": labels, "pairs": pairs, "row": len(rows) - 1}
        alarmed += 1
    return rows, wanted, cache["snippets"]


def evaluate(lmdb, xml, workers, limit_alarmed=0):
    rows, wanted, snippets = prepare(limit_alarmed)
    jobs = [(episode_id, targets, lmdb, xml, snippets)
            for episode_id, targets in wanted.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                row_index = wanted[episode_id][step]["row"]
                rows[row_index].update(result)
            print(f"refined episode {number}/{len(jobs)}", flush=True)
    return rows


def rescore(rows):
    """Reapply the frozen scaling comparison to cached bisection trajectories."""
    for row in rows:
        if "pair_details" not in row:
            row["alarm"] = False
            continue
        early = [pair["early_gaps"] for pair in row["pair_details"]]
        full = [pair["full_gaps"] for pair in row["pair_details"]]
        row.update(result_dict(boundary_refinement_alarm(early, full)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-alarmed", type=int, default=0)
    ap.add_argument("--reuse", default=None,
                    help="rescore pair_details from an existing result without rerunning MuJoCo")
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    args = ap.parse_args()
    if args.reuse:
        rows = rescore(json.loads(Path(args.reuse).read_text())["rows"])
    else:
        with tempfile.TemporaryDirectory(prefix="jenga_boundary_refine_") as temp:
            xml = str(extract_sim(args.sim_archive, temp))
            rows = evaluate(args.lmdb, xml, args.workers, args.limit_alarmed)
    result = {
        "protocol": {
            "calibration_free": True,
            "split": "dev only; test remains untouched",
            "trajectory_source": "MuJoCo ground truth",
            "candidate": "three nearest non-overlapping opposite-response action pairs",
            "refinement": "five midpoint bisections; midpoint assigned by generic trajectory "
                          "proximity to the original response groups",
            "decision": "BIC favours a nonzero gap plateau for a majority of pairs at both "
                        "hold 10 and hold 30",
            "task_information": "none in alarm; Jenga physical classes are grading only",
            "pairs": PAIRS, "refinements": REFINEMENTS,
        },
        "summary": summarise(rows, [1.0]),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["dev"]["1.0"], indent=2))


if __name__ == "__main__":
    main()
