"""Build and diagnose a label-separated Panda upright-push DEV mechanism panel.

``prepare`` uses physical topple labels only to select coverage before any monitor score exists.
``freeze`` binds the exact panel. ``diagnose`` then applies frozen Regime Monitor v0 stage by stage.
This is development evidence for monitor redesign, not a new holdout claim.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,
                                 whole_trajectory_consequence_alarm)
from jenga_regime_monitor_v0_freeze import sha256_file, verify as verify_v0
from jenga_short_held_tails import extract_sim
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT / "results/panda_block_push/dev_panel"
CACHE = OUT / "panel_cache.npz"
PANEL = OUT / "panel.json"
MANIFEST = OUT / "manifest.json"
RESULT = OUT / "v0_stage_diagnostic.json"
HORIZON, HOLD, PROBES, PAIRS, REFINEMENTS = 8, 30, 64, 3, 5
RECORD_HOLDS = (5, 10, 20, 29, 30)
TARGET_COUNTS = {"topple_fork": 20, "safe_centered": 15,
                 "topple_unanimous": 10, "contact_loss_control": 5,
                 "overshoot_control": 6}

PROTOCOL = {
    "id": "panda-upright-push-dev-panel-v0-diagnostic",
    "status": "DEV-only diagnostic; no holdout or universality claim",
    "states": 56,
    "selection": TARGET_COUNTS,
    "selection_information": "trajectory topple labels and planned control family only; no monitor score",
    "topple": "peak block tilt >=45 degrees at any point in H8 plus hold30",
    "fork": "at least 2 of 64 probes upright and at least 2 toppled",
    "probes": "exact frozen Jenga 64xH8 xyz tracking-error snippets at 1x",
    "trajectory": "anonymous single-block pose and velocity padded with invariant rigid-body slots",
    "monitor": "frozen Regime Monitor v0 unchanged: candidate, 3 pairs, 5 bisections, commitment, persistence",
    "task_labels_in_monitor": False,
}


def segment(start, stop, count):
    xyz = np.linspace(np.asarray(start, float), np.asarray(stop, float), count)
    return np.column_stack([xyz, np.ones(count)])


def candidate_specs():
    rows = []; number = 0
    def add(family, z, end_x, y=0., block_y=0., veer_y=0.):
        nonlocal number
        rows.append({"candidate_id": number, "family": family, "z": float(z),
                     "end_x": float(end_x), "contact_y": float(y),
                     "block_y": float(block_y), "veer_y": float(veer_y)})
        number += 1
    for z in np.linspace(.442, .447, 11):
        for end_x in (.55, .56, .57, .58, .59, .60):
            for y in (-.004, 0., .004): add("boundary", z, end_x, y=y)
    for z in (.432, .435, .438, .440):
        for end_x in (.54, .56, .58, .60): add("safe_centered", z, end_x)
    for z in (.448, .452, .456):
        for end_x in (.54, .56, .58, .60): add("topple", z, end_x)
    for block_y in (-.006, 0., .006):
        for veer in (-.075, -.055, .055, .075):
            add("contact_loss", .425, .55, block_y=block_y, veer_y=veer)
    for z in (.425, .432, .438):
        for end_x in (.63, .65, .67): add("overshoot", z, end_x)
    return rows


def _initialise(env, spec):
    env.reset(block_xy=(.535, spec["block_y"]))
    start = env.proprio()[:3]
    behind = np.array([.455, spec["contact_y"] + spec["block_y"], spec["z"]])
    contact = np.array([.500, spec["contact_y"] + spec["block_y"], spec["z"]])
    for action in np.concatenate([segment(start, behind, 20), segment(behind, contact, 15)]):
        env.step(action)
    if spec["family"] == "contact_loss":
        chunk = segment(contact, [spec["end_x"], spec["block_y"] + spec["veer_y"], spec["z"]], HORIZON)
    else:
        chunk = segment(contact, [spec["end_x"], contact[1], spec["z"]], HORIZON)
    return env.snapshot(), chunk


def _run_probe(env, snapshot, chunk, noise, record=False):
    env.restore(snapshot); noisy = np.asarray(chunk, float).copy(); noisy[:, :3] -= noise
    trace = []
    for action in noisy:
        env.step(action)
        if record: trace.append(state_features(env))
    for _ in range(HOLD):
        env.step(noisy[-1])
        if record: trace.append(state_features(env))
    return env.outcome(), np.stack(trace) if record else None


def _grade(job):
    xml, spec, snippets = job; env = PandaBlockPush(xml, render=False)
    try:
        snapshot, chunk = _initialise(env, spec)
        outcomes = [_run_probe(env, snapshot, chunk, noise)[0] for noise in snippets]
        count = sum(row["failure"] for row in outcomes)
        return {**spec, "topple_count": int(count),
                "mixed_topple": bool(2 <= count <= PROBES-2)}
    finally: env.close()


def _select(rows):
    selected = []
    forks = [r for r in rows if r["mixed_topple"]]
    forks.sort(key=lambda r: (abs(r["topple_count"]-PROBES/2), r["candidate_id"]))
    for row in forks[:TARGET_COUNTS["topple_fork"]]: selected.append(("topple_fork", row))

    safe = [r for r in rows if r["family"] == "safe_centered" and r["topple_count"] == 0]
    for row in safe[:TARGET_COUNTS["safe_centered"]]: selected.append(("safe_centered", row))
    topple = [r for r in rows if r["family"] in ("topple", "boundary")
              and r["topple_count"] == PROBES]
    topple.sort(key=lambda r: (r["family"] != "topple", r["candidate_id"]))
    for row in topple[:TARGET_COUNTS["topple_unanimous"]]: selected.append(("topple_unanimous", row))
    contact = [r for r in rows if r["family"] == "contact_loss" and r["topple_count"] == 0]
    for row in contact[:TARGET_COUNTS["contact_loss_control"]]:
        selected.append(("contact_loss_control", row))
    overshoot = [r for r in rows if r["family"] == "overshoot" and r["topple_count"] == 0]
    for row in overshoot[:TARGET_COUNTS["overshoot_control"]]:
        selected.append(("overshoot_control", row))
    observed = {name: sum(cohort == name for cohort, _ in selected) for name in TARGET_COUNTS}
    if observed != TARGET_COUNTS:
        raise RuntimeError(f"candidate pool misses predeclared coverage: {observed}")
    return [{**row, "cohort": cohort, "panel_index": i}
            for i, (cohort, row) in enumerate(selected)]


def _rot6(matrix): return np.asarray(matrix).reshape(3, 3)[:, :2].reshape(-1)


def state_features(env):
    """45-D anonymous one-body state; invariant padding contains no goal or task label."""
    body = env.data.body("push_block"); cvel = np.asarray(body.cvel, float)
    positions = np.r_[body.xpos, np.zeros(6)]
    rotations = np.r_[_rot6(body.xmat), _rot6(np.eye(3)), _rot6(np.eye(3))]
    linear = np.r_[cvel[3:], np.zeros(6)]; angular = np.r_[cvel[:3], np.zeros(6)]
    return np.r_[positions, rotations, linear, angular]


def _collect(job):
    xml, row, snippets = job; env = PandaBlockPush(xml, render=False)
    try:
        snapshot, chunk = _initialise(env, row); traces = []; counts = 0
        for noise in snippets:
            outcome, trace = _run_probe(env, snapshot, chunk, noise, record=True)
            counts += int(outcome["failure"]); traces.append(trace)
        if counts != row["topple_count"]: raise RuntimeError("collection changed physical grade")
        return row, snapshot, chunk, np.stack(traces)
    finally: env.close()


def prepare(archive, workers):
    verify_v0()
    if PANEL.exists() or CACHE.exists(): raise SystemExit("DEV panel already exists")
    snippets = np.load(ROOT / "results/jenga/holdout_stage0_cache.npz")["snippets"]
    with tempfile.TemporaryDirectory(prefix="panda_push_panel_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        specs = candidate_specs(); grades = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_grade, (xml, spec, snippets)) for spec in specs]
            for i, future in enumerate(as_completed(futures), 1):
                grades.append(future.result()); print(f"grade {i}/{len(specs)}", flush=True)
        selected = _select(grades); collected = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_collect, (xml, row, snippets)) for row in selected]
            for i, future in enumerate(as_completed(futures), 1):
                collected.append(future.result()); print(f"collect {i}/{len(selected)}", flush=True)
    collected.sort(key=lambda item: item[0]["panel_index"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, snippets=snippets,
                        snapshots=np.stack([item[1][0] for item in collected]),
                        snapshot_ctrl=np.stack([item[1][1] for item in collected]),
                        snapshot_target=np.stack([item[1][2] for item in collected]),
                        snapshot_gripper=np.asarray([item[1][3] for item in collected]),
                        snapshot_peak_tilt=np.asarray([item[1][4] for item in collected]),
                        chunks=np.stack([item[2] for item in collected]),
                        traces=np.stack([item[3] for item in collected]).astype(np.float32))
    rows = [item[0] for item in collected]
    coverage = {name: {"states": sum(r["cohort"] == name for r in rows),
                       "topple_count_min": min(r["topple_count"] for r in rows if r["cohort"] == name),
                       "topple_count_max": max(r["topple_count"] for r in rows if r["cohort"] == name)}
                for name in TARGET_COUNTS}
    PANEL.write_text(json.dumps({"protocol": PROTOCOL, "coverage": coverage, "rows": rows}, indent=2)+"\n")
    print(json.dumps(coverage, indent=2))


FROZEN_FILES = ("src/systems/panda_block_push.py", "eval/panda_push_dev_panel.py",
                "results/panda_block_push/dev_panel/panel.json",
                "results/panda_block_push/dev_panel/panel_cache.npz",
                "results/jenga/holdout_stage0_cache.npz",
                "results/jenga/regime_monitor_v0/manifest.json")


def _digest(protocol, files):
    value = json.dumps({"protocol": protocol, "files_sha256": files},
                       sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def freeze():
    verify_v0()
    if MANIFEST.exists(): raise SystemExit("DEV panel manifest already exists")
    files = {name: sha256_file(ROOT/name) for name in FROZEN_FILES}
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files, "protocol_sha256": _digest(PROTOCOL, files)}
    MANIFEST.write_text(json.dumps(value, indent=2)+"\n"); return value


def verify():
    verify_v0(); value = json.loads(MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL or tuple(value.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("Panda push DEV protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256_file(ROOT/name) != expected: raise SystemExit(f"Panda push DEV changed: {name}")
    if _digest(PROTOCOL, value["files_sha256"]) != value["protocol_sha256"]:
        raise SystemExit("Panda push DEV digest mismatch")
    return value


def _snapshot(cache, i):
    return (cache["snapshots"][i], cache["snapshot_ctrl"][i], cache["snapshot_target"][i],
            float(cache["snapshot_gripper"][i]), float(cache["snapshot_peak_tilt"][i]))


def _pose_samples(traces):
    indices = [HORIZON+held-1 for held in RECORD_HOLDS]
    pose = np.asarray(traces)[:, indices, :27].copy(); pose[..., :9] /= .05
    return pose


def _projector(reference, dimensions=6):
    flat = np.asarray(reference, float).reshape(len(reference), -1); mean = flat.mean(0)
    centred = flat-mean; _, _, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:min(dimensions, len(axes))]; coordinates = centred@axes.T
    axes = axes[coordinates.std(0) > 1e-10]
    if not len(axes): axes = np.zeros((1, flat.shape[1]))
    scale = max(float(np.sqrt(np.mean(((flat-mean)@axes.T)**2))), 1e-10)
    return lambda values: ((np.asarray(values).reshape(len(values), -1)-mean)@axes.T)/scale


def _reconstruct(noise, pair, sides):
    endpoints = [noise[pair[0]].copy(), noise[pair[1]].copy()]
    for side in sides: endpoints[int(side)] = .5*(endpoints[0]+endpoints[1])
    return np.stack(endpoints)


def _diagnose_state(job):
    xml, i, row, snapshot, chunk, traces, snippets = job
    env = PandaBlockPush(xml, render=False)
    try:
        noise = np.asarray(snippets, float); pose = _pose_samples(traces)
        errors = -noise-(-noise).mean(0, keepdims=True)
        initial = smooth_vs_branch_alarm(errors, pose[:, :2], pose)
        out = {**row, "initial_alarm": bool(initial.alarm), "boundary_refined_alarm": False,
               "alarm": False, "initial_evidence": result_dict(initial)}
        if not initial.alarm: return out
        x, labels = action_branch_partition(errors, pose[:, :2], pose)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        ep, fp = _projector(pose[:, :2]), _projector(pose)
        original_e, original_f = ep(pose[:, :2]), fp(pose)
        centroids = np.stack([original_f[labels == group].mean(0) for group in (0, 1)])
        early_gaps, full_gaps, details = [], [], []
        for left, right in pairs:
            endpoint_noise = [noise[left].copy(), noise[right].copy()]
            endpoint_pose = [pose[left].copy(), pose[right].copy()]
            eg = [float(np.linalg.norm(original_e[left]-original_e[right]))]
            fg = [float(np.linalg.norm(original_f[left]-original_f[right]))]; sides = []
            for _ in range(REFINEMENTS):
                midpoint = .5*(endpoint_noise[0]+endpoint_noise[1])
                _, trace = _run_probe(env, snapshot, chunk, midpoint, record=True)
                mp = _pose_samples(trace[None])[0]; projected = fp(mp[None])[0]
                side = int(np.argmin(np.linalg.norm(centroids-projected[None], axis=1)))
                endpoint_noise[side] = midpoint; endpoint_pose[side] = mp; sides.append(side)
                ee = ep(np.stack([p[:2] for p in endpoint_pose])); ff = fp(np.stack(endpoint_pose))
                eg.append(float(np.linalg.norm(ee[0]-ee[1]))); fg.append(float(np.linalg.norm(ff[0]-ff[1])))
            early_gaps.append(eg); full_gaps.append(fg)
            details.append({"probe_indices": [int(left), int(right)], "midpoint_sides": sides})
        boundary = boundary_refinement_alarm(early_gaps, full_gaps)
        endpoints = np.stack([_reconstruct(noise, pair, detail["midpoint_sides"])
                              for pair, detail in zip(pairs, details)])
        dense = np.stack([[_run_probe(env, snapshot, chunk, endpoint, record=True)[1]
                           for endpoint in endpoint_pair] for endpoint_pair in endpoints])
        evidence = whole_trajectory_consequence_alarm(
            dense, endpoints[:, 0]-endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(evidence)})
        return out
    finally: env.close()


def summarise(rows):
    out = {}
    for cohort in TARGET_COUNTS:
        selected = [r for r in rows if r["cohort"] == cohort]
        out[cohort] = {"states": len(selected),
                       "initial_candidates": sum(r["initial_alarm"] for r in selected),
                       "refined_boundaries": sum(r["boundary_refined_alarm"] for r in selected),
                       "commitment_majority": sum(r.get("commitment_pairs", 0) >= 2 for r in selected),
                       "persistence_majority": sum(r.get("persistence_pairs", 0) >= 2 for r in selected),
                       "final_alarms": sum(r["alarm"] for r in selected)}
    return out


def diagnose(archive, workers):
    manifest = verify()
    if RESULT.exists(): raise SystemExit("DEV diagnostic already exists")
    panel = json.loads(PANEL.read_text()); cache = np.load(CACHE, allow_pickle=False)
    with tempfile.TemporaryDirectory(prefix="panda_push_diagnose_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        jobs = [(xml, i, row, _snapshot(cache, i), cache["chunks"][i], cache["traces"][i],
                 cache["snippets"]) for i, row in enumerate(panel["rows"])]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_diagnose_state, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                rows.append(future.result()); print(f"diagnose {i}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: r["panel_index"]); summary = summarise(rows)
    value = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": summary, "rows": rows}
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "diagnose"):
        child = sub.add_parser(name); child.add_argument("--workers", type=int, default=8)
        child.add_argument("--sim-archive", default=str(ROOT/"vendor/panda_express_sim.tar"))
    sub.add_parser("freeze"); sub.add_parser("verify"); args = parser.parse_args()
    if args.command == "prepare": prepare(args.sim_archive, args.workers)
    elif args.command == "freeze": print(freeze()["protocol_sha256"])
    elif args.command == "verify": print(verify()["protocol_sha256"])
    else: diagnose(args.sim_archive, args.workers)


if __name__ == "__main__": main()
