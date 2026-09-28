"""Prospectively frozen intervention-utility experiment for Regime Monitor v0.

One hundred recorded action trajectories are replayed under fresh simulator reset seeds. Four
matched arms compare no monitor, one-step reobservation, a label-blind robust-action wrapper, and a
privileged local grading oracle. Regime Monitor v0 is verified before freeze, verification and run.
The canonical result cannot be overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, nearest_cross_branch_pairs,  # noqa: E402
                                   smooth_vs_branch_alarm)
from consequence_monitor import whole_trajectory_consequence_alarm  # noqa: E402
from intervention_policy import (ACTION_SCALES, choose_oracle_candidate,  # noqa: E402
                                 choose_robust_candidate, hold_action,
                                 scaled_action_candidates)
from jenga_action_boundary_refine import PAIRS, _refine_state  # noqa: E402
from jenga_action_branch import pose_features  # noqa: E402
from jenga_bench import BENCH_FILE, verify as verify_benchmark  # noqa: E402
from jenga_generic_regime_audit import reconstruct_endpoints, rollout_trace  # noqa: E402
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_regime_monitor_v0_freeze import (sha256_file, verify as verify_v0)  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT, rollout  # noqa: E402


EXPERIMENT_DIR = ROOT / "results/jenga/intervention_v0"
MANIFEST = EXPERIMENT_DIR / "manifest.json"
OUTPUT = EXPERIMENT_DIR / "ground_truth_100.json"
EPISODES = 100
RESET_BASE = 2000
HOLD_STEPS = 30
ARMS = ("no_monitor", "reobserve", "wrapper", "oracle")
PROTOCOL = {
    "id": "jenga-intervention-v0",
    "detector": "frozen Regime Monitor v0, unchanged",
    "episodes": EPISODES,
    "episode_source": "all 100 recorded absolute-action trajectories",
    "fresh_configuration": "simulator reset seed = 2000 + episode id",
    "decision_points": "non-overlapping H=8 chunks beginning at action 2",
    "arms": list(ARMS),
    "reobserve": "one current-pose/current-gripper hold control step, then rerun v0 once",
    "persistent_alarm_candidates": list(ACTION_SCALES),
    "candidate_definition": "contract every intended xyz target toward current EE xyz; preserve "
                            "intended gripper except scale 0 holds current gripper",
    "wrapper_choice": "fewest v0 consequential pairs; ties closest to intended action",
    "oracle_choice": "avoid local new neighbor topple over H=8+hold30; ties closest to intended",
    "intervention_cap": "at most one reobservation and one modified chunk per decision point",
    "grading_only": "neighbor tilt >=45 deg; pick proxy is middle lift >=2.5 cm and lateral "
                    "motion >=2 cm after the first ten policy actions",
    "labels_in_policy": False,
}
FROZEN_FILES = (
    "src/intervention_policy.py",
    "eval/jenga_intervention_v0.py",
    "results/jenga/regime_monitor_v0/manifest.json",
    "results/jenga/bench/manifest.json",
    "results/jenga/bench/jenga_bench.npz",
    "data/jenga/jenga_single_100.lmdb/data.mdb",
    "vendor/panda_express_sim.tar",
)


def _digest(protocol, files):
    value = json.dumps({"protocol": protocol, "files_sha256": files},
                       sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def freeze():
    verify_v0(); verify_benchmark()
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} already exists; create a new named protocol version")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing files: {missing}")
    files = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    manifest = {"protocol": PROTOCOL,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_sha256": _digest(PROTOCOL, files), "files_sha256": files}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify():
    verify_v0(); verify_benchmark()
    if not MANIFEST.is_file():
        raise SystemExit("intervention v0 is not frozen")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("protocol") != PROTOCOL or tuple(manifest.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("intervention protocol differs from its manifest")
    changed = []
    for name, expected in manifest["files_sha256"].items():
        actual = sha256_file(ROOT / name) if (ROOT / name).is_file() else None
        if actual != expected:
            changed.append((name, expected, actual))
    if changed:
        raise SystemExit("intervention freeze mismatch:\n" + "\n".join(
            f"  {name}: expected {expected}, now {actual}" for name, expected, actual in changed))
    if _digest(PROTOCOL, manifest["files_sha256"]) != manifest.get("protocol_sha256"):
        raise SystemExit("intervention protocol digest is invalid")
    return manifest


def _action_errors(probe_chunks):
    xyz = np.asarray(probe_chunks, float)[..., :3]
    return xyz - xyz.mean(0, keepdims=True)


def monitor_action(sim, snapshot, chunk, snippets):
    """Apply frozen v0 to an arbitrary current simulator state and intended H=8 chunk."""
    chunk = np.asarray(chunk, np.float32)
    probe_chunks = np.repeat(chunk[None], len(snippets), axis=0)
    probe_chunks[:, :, :3] -= snippets
    start_pose = all_block_pose(sim)
    start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
    simulated = [rollout(sim, snapshot, chunk, noise, start_pose, start_tilt)[0]
                 for noise in snippets]
    pose = np.asarray(simulated, np.float32)
    features = pose_features(pose)
    early_stop = RECORD_AT.index(10) + 1
    errors = _action_errors(probe_chunks)
    initial = smooth_vs_branch_alarm(errors, features[:, :early_stop], features)
    result = {"initial_alarm": bool(initial.alarm), "boundary_alarm": False,
              "alarm": False, "consequential_pairs": 0}
    if not initial.alarm:
        sim.restore(snapshot)
        return result
    coordinates, labels = action_branch_partition(errors, features[:, :early_stop], features)
    pairs = nearest_cross_branch_pairs(coordinates, labels, PAIRS)
    boundary, pair_details = _refine_state(
        sim, snapshot, chunk, snippets, pose, labels, pairs, start_pose, start_tilt)
    endpoints = reconstruct_endpoints(snippets, pair_details)
    traces = np.stack([[rollout_trace(sim, snapshot, chunk, noise)
                        for noise in pair] for pair in endpoints])
    evidence = whole_trajectory_consequence_alarm(
        traces[:, :, :, :45], endpoints[:, 0] - endpoints[:, 1],
        boundary["early_delta_bic"], boundary["full_delta_bic"])
    sim.restore(snapshot)
    return {"initial_alarm": True, "boundary_alarm": bool(boundary["alarm"]),
            "alarm": bool(evidence.alarm),
            "consequential_pairs": int(evidence.consequential_pairs),
            "boundary_pairs": int(evidence.boundary_pairs),
            "commitment_pairs": int(evidence.commitment_pairs),
            "persistence_pairs": int(evidence.persistence_pairs)}


def _monitor_cache_key(snapshot, chunk):
    """Exact deterministic identity; cache reuse cannot merge merely similar states."""
    digest = hashlib.sha256()
    for value in snapshot:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    actions = np.ascontiguousarray(np.asarray(chunk, np.float32))
    digest.update(actions.tobytes())
    return digest.hexdigest()


def cached_monitor_action(cache, sim, snapshot, chunk, snippets):
    key = _monitor_cache_key(snapshot, chunk)
    if key not in cache:
        cache[key] = monitor_action(sim, snapshot, chunk, snippets)
    else:
        sim.restore(snapshot)
    return dict(cache[key])


def current_action(sim):
    prop = sim.proprio()
    return np.asarray(prop, np.float32)


def physical_candidate_grade(sim, snapshot, chunk):
    """Privileged local grade used only by the oracle arm."""
    sim.restore(snapshot)
    start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
    peak = start_tilt.copy()
    for action in chunk:
        sim.execute(action)
        peak = np.maximum(peak, sim.block_diagnostics()["tilt"][1:])
    for _ in range(HOLD_STEPS):
        sim.execute(chunk[-1])
        peak = np.maximum(peak, sim.block_diagnostics()["tilt"][1:])
    eligible = start_tilt < 45.0
    failed = bool(np.any((peak >= 45.0) & eligible))
    sim.restore(snapshot)
    return failed, float(np.max(peak))


def _record(trace, sim, original_index):
    state = sim.block_diagnostics()
    trace["positions"].append(state["position"].copy())
    trace["tilts"].append(state["tilt"].copy())
    trace["original_indices"].append(int(original_index))


def episode_outcome(trace, reset_position):
    positions = np.asarray(trace["positions"])
    tilts = np.asarray(trace["tilts"])
    indices = np.asarray(trace["original_indices"])
    baseline_rows = np.flatnonzero(indices >= 9)
    baseline_index = int(baseline_rows[0]) if len(baseline_rows) else 0
    baseline = positions[baseline_index, 0] if len(positions) else reset_position[0]
    after = positions[baseline_index:, 0] if len(positions) else reset_position[None, 0]
    lift = float(np.max(after[:, 2] - baseline[2]))
    lateral = float(np.max(np.linalg.norm(after[:, :2] - baseline[:2], axis=1)))
    neighbor_peak = float(np.max(tilts[:, 1:])) if len(tilts) else 0.0
    pick = bool(lift >= .025 and lateral >= .02)
    failure = bool(neighbor_peak >= 45.0)
    return {"neighbor_failure": failure, "pick_success": pick,
            "safe_completion": bool(pick and not failure),
            "middle_peak_lift_m": lift, "middle_peak_lateral_m": lateral,
            "neighbor_peak_tilt_deg": neighbor_peak}


def run_arm(sim, episode, arm, snippets, reset_seed, monitor_cache=None):
    monitor_cache = {} if monitor_cache is None else monitor_cache
    sim.reset(reset_seed)
    reset_position = sim.block_diagnostics()["position"].copy()
    trace = {"positions": [], "tilts": [], "original_indices": []}
    decisions = []
    i = 0
    while i < len(episode.actions):
        is_decision = i >= 2 and (i - 2) % HORIZON == 0 and i + HORIZON <= len(episode.actions)
        if not is_decision:
            sim.execute(episode.actions[i]); _record(trace, sim, i); i += 1
            continue
        original = np.asarray(episode.actions[i:i + HORIZON], np.float32)
        chosen = original
        decision = {"chunk_start": i, "initial_alarm": False, "persistent_alarm": False,
                    "reobserved": False, "chosen_scale": 1.0}
        if arm != "no_monitor":
            before = sim.snapshot()
            first = cached_monitor_action(monitor_cache, sim, before, original, snippets)
            decision.update({"initial_alarm": bool(first["alarm"]),
                             "initial_consequential_pairs": first["consequential_pairs"]})
            if first["alarm"]:
                held = hold_action(current_action(sim))
                sim.execute(held); _record(trace, sim, i - 1)
                decision["reobserved"] = True
                after_hold = sim.snapshot()
                second = cached_monitor_action(monitor_cache, sim, after_hold, original, snippets)
                decision.update({"persistent_alarm": bool(second["alarm"]),
                                 "persistent_consequential_pairs": second["consequential_pairs"]})
                if second["alarm"] and arm in ("wrapper", "oracle"):
                    candidates = scaled_action_candidates(original, current_action(sim))
                    if arm == "wrapper":
                        evidence = [second]
                        evidence.extend(cached_monitor_action(
                            monitor_cache, sim, after_hold, candidate, snippets)
                            for candidate in candidates[1:])
                        selection = choose_robust_candidate(evidence)
                        index = selection.index
                        decision["candidate_consequential_pairs"] = [
                            int(row["consequential_pairs"]) for row in evidence]
                    else:
                        grades = [physical_candidate_grade(sim, after_hold, candidate)
                                  for candidate in candidates]
                        index = choose_oracle_candidate(grades)
                        decision["candidate_neighbor_topples"] = [bool(row[0]) for row in grades]
                        decision["candidate_peak_tilt_deg"] = [float(row[1]) for row in grades]
                    chosen = candidates[index]
                    decision["chosen_scale"] = float(ACTION_SCALES[index])
                sim.restore(after_hold)
        for offset, action in enumerate(chosen):
            sim.execute(action); _record(trace, sim, i + offset)
        decisions.append(decision)
        i += HORIZON
    outcome = episode_outcome(trace, reset_position)
    outcome.update({
        "decision_points": len(decisions),
        "initial_alarms": sum(row["initial_alarm"] for row in decisions),
        "persistent_alarms": sum(row["persistent_alarm"] for row in decisions),
        "reobservations": sum(row["reobserved"] for row in decisions),
        "modified_chunks": sum(row["chosen_scale"] < 1 for row in decisions),
        "added_control_steps": sum(row["reobserved"] for row in decisions),
        "mean_chosen_scale_on_modification": (float(np.mean([
            row["chosen_scale"] for row in decisions if row["chosen_scale"] < 1]))
            if any(row["chosen_scale"] < 1 for row in decisions) else None),
    })
    return {"arm": arm, **outcome, "decisions": decisions}


def _run_episode(job):
    episode_id, lmdb, xml, snippets = job
    replay = JengaReplay(lmdb); episode = replay.episode(episode_id); replay.close()
    sim = DirectJengaSim(xml)
    try:
        monitor_cache = {}
        arms = {arm: run_arm(sim, episode, arm, snippets, RESET_BASE + int(episode_id),
                             monitor_cache)
                for arm in ARMS}
    finally:
        sim.close()
    return {"episode_id": str(episode_id), "reset_seed": RESET_BASE + int(episode_id),
            "arms": arms}


def summarise(rows):
    summary = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        summary[arm] = {
            "episodes": len(values),
            "neighbor_failures": sum(row["neighbor_failure"] for row in values),
            "pick_successes": sum(row["pick_success"] for row in values),
            "safe_completions": sum(row["safe_completion"] for row in values),
            "episodes_intervened": sum(row["reobservations"] > 0 for row in values),
            "reobservations": sum(row["reobservations"] for row in values),
            "persistent_alarms": sum(row["persistent_alarms"] for row in values),
            "modified_chunks": sum(row["modified_chunks"] for row in values),
            "added_control_steps": sum(row["added_control_steps"] for row in values),
        }
    baseline = {row["episode_id"]: row["arms"]["no_monitor"] for row in rows}
    for arm in ARMS[1:]:
        values = {row["episode_id"]: row["arms"][arm] for row in rows}
        summary[arm]["baseline_failures_prevented"] = sum(
            baseline[key]["neighbor_failure"] and not values[key]["neighbor_failure"]
            for key in baseline)
        summary[arm]["new_failures_vs_baseline"] = sum(
            not baseline[key]["neighbor_failure"] and values[key]["neighbor_failure"]
            for key in baseline)
    return summary


def run(args):
    manifest = verify()
    if Path(args.output).resolve() != OUTPUT.resolve():
        raise SystemExit(f"canonical intervention result must be written to {OUTPUT}")
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite immutable intervention result {OUTPUT}")
    replay = JengaReplay(args.lmdb)
    try:
        ids = replay.episode_ids
    finally:
        replay.close()
    if len(ids) != EPISODES:
        raise SystemExit(f"frozen protocol requires exactly {EPISODES} episodes, found {len(ids)}")
    snippets = np.load(BENCH_FILE, allow_pickle=False)["snippets"]
    with tempfile.TemporaryDirectory(prefix="jenga_intervention_v0_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        jobs = [(episode_id, args.lmdb, xml, snippets) for episode_id in ids]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
            for number, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                print(f"intervention episode {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: int(row["episode_id"]))
    result = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "summary": summarise(rows), "rows": rows}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("verify")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    run_parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(); print(f"frozen: {result['protocol_sha256']}")
    elif args.command == "verify":
        result = verify(); print(f"intervention v0 intact: {result['protocol_sha256']}")
    else:
        run(args)


if __name__ == "__main__":
    main()
