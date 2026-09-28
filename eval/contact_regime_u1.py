"""Prospective ground-truth universality test for frozen Regime Monitor v0.

The benchmark is built and labelled before any monitor output is computed.  ``prepare`` records
the 64 counterfactual futures and grading-only mechanism labels, ``freeze`` checksums that cache,
and ``run`` applies the byte-frozen Jenga v0 implementation without task-specific inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,
                                 whole_trajectory_consequence_alarm)
from jenga_regime_monitor_v0_freeze import sha256_file, verify as verify_v0
from systems.contact_benchmarks import InsertionBenchmark, PushingBenchmark


OUT = ROOT / "results/contact_benchmarks/u1"
CACHE = OUT / "mechanism_panel.npz"
PANEL = OUT / "mechanism_panel.json"
MANIFEST = OUT / "manifest.json"
RESULT = OUT / "ground_truth_detector.json"
HORIZON, HOLD, PROBES, PAIRS, REFINEMENTS = 8, 30, 64, 3, 5
RECORD_HOLDS = (5, 10, 20, 29, 30)
EPISODES_PER_TASK = 30
CHUNK_STARTS = (24, 32, 40, 48, 56, 64, 72, 80)

PROTOCOL = {
    "id": "contact-regime-u1",
    "purpose": "prospective ground-truth cross-task universality test",
    "tasks": ["pushing", "insertion"],
    "panel": "30 deterministic episodes x 8 non-overlapping H=8 states per task",
    "split": "episodes 0-14 DEV, 15-29 untouched TEST",
    "execution_error": "the exact frozen 64 x H=8 Jenga translation tracking snippets at 1x",
    "pushing_map": "snippet xyz -> commanded planar xy (z discarded)",
    "insertion_map": "snippet xyz -> xyz plus yaw=snippet_x/(20*sqrt(2) mm), so yaw has the same corner displacement",
    "trajectory": "three anonymous rigid-body slots; position/rotation-6D/linear/angular velocity",
    "candidate_samples": "hold steps 5,10,20,29,30; early evidence uses 5,10",
    "monitor": "frozen Regime Monitor v0 unchanged: 64 probes, H8, hold30, 3 pairs, 5 bisections",
    "grading": "task success/failure/mode labels are attached before monitor scoring and never enter it",
    "primary_gate": ">=50% consequential task-outcome-fork recall in each task and >=25 percentage-point separation from unanimous quiet",
    "primary_execution_error_scale": 1.0,
}


def _digest(protocol, files):
    value = json.dumps({"protocol": protocol, "files_sha256": files},
                       sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _episode_controls(task, episode):
    rng = np.random.default_rng(91000 + episode)
    if task == "pushing":
        object_y = float(rng.uniform(-.05, .05))
        target_y = float(np.clip(object_y + rng.uniform(-.012, .012), -.12, .12))
        # A prospectively tiled edge-transition panel.  A grading-only pilot located the physical
        # transition in [0.7195, 0.7255]; the monitor was not run during that coverage check.
        end_x = float(np.linspace(.7195, .7255, 5)[episode % 5])
        controls = np.column_stack([np.linspace(0., end_x, 96),
                                    np.linspace(0., target_y, 96)])
        return {"object_y": object_y, "target_y": target_y, "end_x": end_x}, controls
    lateral = rng.uniform(-.014, .014, size=2)
    yaw = float(rng.uniform(-.12, .12))
    z_end = float(rng.uniform(-.155, -.135))
    phase = np.linspace(0., 1., 96)
    controls = np.column_stack([phase * lateral[0], phase * lateral[1],
                                phase * z_end, phase * yaw])
    return {"lateral_x": float(lateral[0]), "lateral_y": float(lateral[1]),
            "yaw": yaw, "z_end": z_end}, controls


def _reset(env, task, params):
    if task == "pushing":
        env.reset(params["object_y"])
    else:
        env.reset()


def _rot6(matrix):
    return np.asarray(matrix, float).reshape(3, 3)[:, :2].reshape(-1)


def _body(env, name):
    body = env.data.body(name)
    cvel = np.asarray(body.cvel, float)
    return (np.asarray(body.xpos, float).copy(), _rot6(body.xmat),
            cvel[3:].copy(), cvel[:3].copy())


def _fixed_geom(env, name):
    geom = env.data.geom(name)
    return (np.asarray(geom.xpos, float).copy(), _rot6(geom.xmat),
            np.zeros(3), np.zeros(3))


def state_features(env, task):
    """Anonymous 45-D three-rigid-body state expected by v0; no outcomes or contacts."""
    if task == "pushing":
        bodies = (_body(env, "pusher"), _body(env, "object"), _fixed_geom(env, "table"))
    else:
        bodies = (_body(env, "peg"), _fixed_geom(env, "socket_left"),
                  _fixed_geom(env, "socket_right"))
    return np.concatenate([np.concatenate([b[i] for b in bodies]) for i in range(4)])


def map_noise(task, snippets, scale=1.0):
    noise = np.asarray(snippets, float) * float(scale)
    if task == "pushing":
        return noise[..., :2]
    yaw = noise[..., :1] / (.020 * np.sqrt(2.))
    return np.concatenate([noise, yaw], axis=-1)


def _rollout(env, task, snapshot, chunk, noise):
    env.restore(snapshot)
    traces = []
    commands = np.asarray(chunk, float) - np.asarray(noise, float)
    for command in commands:
        env.step(command); traces.append(state_features(env, task))
    for _ in range(HOLD):
        env.step(commands[-1]); traces.append(state_features(env, task))
    return np.stack(traces), env.outcome()


def _label(outcomes):
    failure = np.asarray([o.failure for o in outcomes], bool)
    success = np.asarray([o.success for o in outcomes], bool)
    modes = [o.mode for o in outcomes]
    counts = {mode: modes.count(mode) for mode in sorted(set(modes))}
    if (2 <= int(failure.sum()) <= len(failure) - 2
            or 2 <= int(success.sum()) <= len(success) - 2):
        cohort = "consequential_fork"
    elif len(counts) > 1 and sorted(counts.values())[-2] >= 2:
        cohort = "mode_fork"
    else:
        cohort = "unanimous_quiet"
    return {"cohort": cohort, "failure_count": int(failure.sum()),
            "success_count": int(success.sum()), "mode_counts": counts}


def _prepare_state(job):
    task, episode, chunk_start, snippets = job
    env = PushingBenchmark(width=64, height=64) if task == "pushing" else InsertionBenchmark(width=64, height=64)
    try:
        params, controls = _episode_controls(task, episode)
        _reset(env, task, params)
        for command in controls[:chunk_start]:
            env.step(command)
        snapshot = env.snapshot()
        chunk = controls[chunk_start:chunk_start + HORIZON]
        traces, outcomes = [], []
        for noise in map_noise(task, snippets, 1.0):
            trace, outcome = _rollout(env, task, snapshot, chunk, noise)
            traces.append(trace); outcomes.append(outcome)
        row = {"task": task, "episode": episode, "chunk_start": chunk_start,
               "split": "dev" if episode < EPISODES_PER_TASK // 2 else "test",
               "params": params, **_label(outcomes)}
        return row, snapshot, chunk, np.stack(traces)
    finally:
        env.close()


def prepare(workers):
    verify_v0()
    if CACHE.exists() or PANEL.exists():
        raise SystemExit("prospective panel already exists; refusing to overwrite")
    snippets = np.load(ROOT / "results/jenga/holdout_stage0_cache.npz")["snippets"]
    if snippets.shape != (PROBES, HORIZON, 3):
        raise SystemExit(f"unexpected frozen snippet shape {snippets.shape}")
    jobs = [(task, episode, chunk, snippets) for task in ("pushing", "insertion")
            for episode in range(EPISODES_PER_TASK) for chunk in CHUNK_STARTS]
    rows, arrays = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_prepare_state, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            row, snapshot, chunk, traces = future.result()
            arrays.append((row, snapshot, chunk, traces))
            print(f"panel state {number}/{len(jobs)}", flush=True)
    arrays.sort(key=lambda item: (item[0]["task"], item[0]["episode"], item[0]["chunk_start"]))
    rows = [item[0] for item in arrays]
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_sizes = np.asarray([len(x[1]) for x in arrays], np.int16)
    snapshots = np.zeros((len(arrays), int(snapshot_sizes.max())), np.float64)
    for i, (_, snapshot, _, _) in enumerate(arrays):
        snapshots[i, :len(snapshot)] = snapshot
    action_dims = np.asarray([x[2].shape[1] for x in arrays], np.int8)
    chunks = np.zeros((len(arrays), HORIZON, int(action_dims.max())), np.float64)
    for i, (_, _, chunk, _) in enumerate(arrays):
        chunks[i, :, :chunk.shape[1]] = chunk
    np.savez_compressed(CACHE, snippets=snippets,
                        snapshots=snapshots, snapshot_sizes=snapshot_sizes,
                        chunks=chunks, action_dims=action_dims,
                        traces=np.stack([x[3] for x in arrays]))
    coverage = summarise_labels(rows)
    PANEL.write_text(json.dumps({"protocol": PROTOCOL, "coverage": coverage, "rows": rows},
                                indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


def summarise_labels(rows):
    out = {}
    for task in ("pushing", "insertion"):
        out[task] = {}
        for split in ("dev", "test"):
            selected = [r for r in rows if r["task"] == task and r["split"] == split]
            out[task][split] = {name: sum(r["cohort"] == name for r in selected)
                                for name in ("consequential_fork", "mode_fork",
                                             "unanimous_quiet")}
    return out


FROZEN_FILES = ("src/systems/contact_benchmarks.py", "eval/contact_regime_u1.py",
                "results/contact_benchmarks/u1/mechanism_panel.npz",
                "results/contact_benchmarks/u1/mechanism_panel.json",
                "results/jenga/regime_monitor_v0/manifest.json")


def freeze():
    verify_v0()
    if MANIFEST.exists():
        raise SystemExit("U1 manifest already exists; refusing to overwrite")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing prospective inputs: {missing}")
    files = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    manifest = {"protocol": PROTOCOL,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files_sha256": files, "protocol_sha256": _digest(PROTOCOL, files)}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify():
    verify_v0()
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("protocol") != PROTOCOL or tuple(manifest.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("U1 protocol differs from manifest")
    changed = [(name, expected, sha256_file(ROOT / name) if (ROOT / name).is_file() else None)
               for name, expected in manifest["files_sha256"].items()
               if (sha256_file(ROOT / name) if (ROOT / name).is_file() else None) != expected]
    if changed:
        raise SystemExit("U1 freeze mismatch:\n" + "\n".join(map(str, changed)))
    if _digest(PROTOCOL, manifest["files_sha256"]) != manifest["protocol_sha256"]:
        raise SystemExit("U1 protocol digest mismatch")
    return manifest


def _pose_samples(traces):
    indices = [HORIZON + held - 1 for held in RECORD_HOLDS]
    pose = np.asarray(traces)[:, indices, :27].copy()
    pose[..., :9] /= .05
    return pose


def _projector(reference, dimensions=6):
    flat = np.asarray(reference, float).reshape(len(reference), -1)
    mean = flat.mean(0); centred = flat - mean
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:min(dimensions, len(axes))]
    coordinates = centred @ axes.T
    axes = axes[coordinates.std(0) > 1e-10]
    if not len(axes):
        axes = np.zeros((1, flat.shape[1]))
    scale = max(float(np.sqrt(np.mean(((flat - mean) @ axes.T) ** 2))), 1e-10)
    return lambda values: ((np.asarray(values).reshape(len(values), -1) - mean) @ axes.T) / scale


def _evaluate_state(job):
    index, row, snapshot, chunk, traces, snippets = job
    task = row["task"]
    env = PushingBenchmark(width=64, height=64) if task == "pushing" else InsertionBenchmark(width=64, height=64)
    try:
        noise = map_noise(task, snippets)
        pose = _pose_samples(traces)
        errors = -noise - (-noise).mean(0, keepdims=True)
        initial = smooth_vs_branch_alarm(errors, pose[:, :2], pose)
        result = {**row, "index": index, "initial_alarm": bool(initial.alarm),
                  "boundary_refined_alarm": False, "alarm": False,
                  "initial_evidence": result_dict(initial)}
        if not initial.alarm:
            return result
        x, labels = action_branch_partition(errors, pose[:, :2], pose)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        early_project, full_project = _projector(pose[:, :2]), _projector(pose)
        original_early, original_full = early_project(pose[:, :2]), full_project(pose)
        centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
        early_gaps, full_gaps, endpoint_rows = [], [], []
        for left, right in pairs:
            endpoint_noise = [noise[left].copy(), noise[right].copy()]
            endpoint_pose = [pose[left].copy(), pose[right].copy()]
            pair_early = [float(np.linalg.norm(original_early[left] - original_early[right]))]
            pair_full = [float(np.linalg.norm(original_full[left] - original_full[right]))]
            midpoint_sides = []
            for _ in range(REFINEMENTS):
                midpoint_noise = .5 * (endpoint_noise[0] + endpoint_noise[1])
                midpoint_trace, _ = _rollout(env, task, snapshot, chunk, midpoint_noise)
                midpoint_pose = _pose_samples(midpoint_trace[None])[0]
                projected = full_project(midpoint_pose[None])[0]
                side = int(np.argmin(np.linalg.norm(centroids - projected[None], axis=1)))
                endpoint_noise[side] = midpoint_noise; endpoint_pose[side] = midpoint_pose
                ep_early = early_project(np.stack([p[:2] for p in endpoint_pose]))
                ep_full = full_project(np.stack(endpoint_pose))
                pair_early.append(float(np.linalg.norm(ep_early[0] - ep_early[1])))
                pair_full.append(float(np.linalg.norm(ep_full[0] - ep_full[1])))
                midpoint_sides.append(side)
            early_gaps.append(pair_early); full_gaps.append(pair_full)
            endpoint_rows.append({"probe_indices": [int(left), int(right)],
                                  "midpoint_sides": midpoint_sides})
        boundary = boundary_refinement_alarm(early_gaps, full_gaps)
        dense = np.stack([[_rollout(env, task, snapshot, chunk, endpoint)[0]
                           for endpoint in endpoint_noise_pair]
                          for endpoint_noise_pair in [
                              _reconstruct(noise, pair, details["midpoint_sides"])
                              for pair, details in zip(pairs, endpoint_rows)]])
        endpoints = np.stack([_reconstruct(noise, pair, details["midpoint_sides"])
                              for pair, details in zip(pairs, endpoint_rows)])
        evidence = whole_trajectory_consequence_alarm(
            dense, endpoints[:, 0] - endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        result.update({"boundary_refined_alarm": bool(boundary.alarm),
                       "boundary_evidence": result_dict(boundary),
                       "pair_details": endpoint_rows,
                       **consequence_result_dict(evidence)})
        return result
    finally:
        env.close()


def _reconstruct(noise, pair, sides):
    endpoints = [noise[pair[0]].copy(), noise[pair[1]].copy()]
    for side in sides:
        endpoints[int(side)] = .5 * (endpoints[0] + endpoints[1])
    return np.stack(endpoints)


def summarise_results(rows):
    out = {}
    for task in ("pushing", "insertion"):
        out[task] = {}
        for split in ("dev", "test"):
            part = [r for r in rows if r["task"] == task and r["split"] == split]
            cohorts = {}
            for name in ("consequential_fork", "mode_fork", "unanimous_quiet"):
                selected = [r for r in part if r["cohort"] == name]
                cohorts[name] = {"states": len(selected),
                                 "initial_candidates": sum(r["initial_alarm"] for r in selected),
                                 "refined_boundaries": sum(r["boundary_refined_alarm"] for r in selected),
                                 "alarms": sum(r["alarm"] for r in selected),
                                 "rate": (float(np.mean([r["alarm"] for r in selected]))
                                          if selected else None)}
            consequence = cohorts["consequential_fork"]; quiet = cohorts["unanimous_quiet"]
            separation = ((consequence["rate"] - quiet["rate"])
                          if consequence["rate"] is not None and quiet["rate"] is not None else None)
            out[task][split] = {"cohorts": cohorts, "failure_quiet_separation": separation,
                                "gate_passed": bool(consequence["states"] and quiet["states"] and
                                                    consequence["rate"] >= .5 and separation >= .25)}
    return out


def run(workers):
    manifest = verify()
    if RESULT.exists():
        raise SystemExit("canonical U1 result exists; refusing to overwrite")
    panel = json.loads(PANEL.read_text())
    cache = np.load(CACHE, allow_pickle=False)
    jobs = [(i, row, cache["snapshots"][i, :cache["snapshot_sizes"][i]],
             cache["chunks"][i, :, :cache["action_dims"][i]], cache["traces"][i],
             cache["snippets"]) for i, row in enumerate(panel["rows"])]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_state, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            print(f"v0 state {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: r["index"])
    summary = summarise_results(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "summary": summary, "rows": rows}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        child = sub.add_parser(command); child.add_argument("--workers", type=int, default=8)
    sub.add_parser("freeze"); sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.workers)
    elif args.command == "freeze":
        print(json.dumps(freeze(), indent=2))
    elif args.command == "verify":
        print(f"U1 intact: {verify()['protocol_sha256']}")
    else: run(args.workers)


if __name__ == "__main__":
    main()
