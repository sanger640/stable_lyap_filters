"""Replay frozen intervention-v0 decisions and measure control time to successful picking.

No monitor is rerun and no action is reselected. The immutable decision log is deterministically
replayed under the same reset seeds, adding timestamps to the already graded outcomes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from intervention_policy import ACTION_SCALES, hold_action, scaled_action_candidates  # noqa: E402
from jenga_intervention_v0 import (ARMS, OUTPUT as RESULT, RESET_BASE, current_action,  # noqa: E402
                                   sha256_file, verify)
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402


EXPECTED_RESULT_SHA256 = "b0695221f4e0d56133fb64cd6e04da796c8c7466e2188d4b1f42504c6ec06b42"
OUTPUT = ROOT / "results/jenga/intervention_v0/time_to_success.json"
CONTROL_PERIOD_S = 0.1


def _record(trace, sim, original_index):
    state = sim.block_diagnostics()
    trace["positions"].append(state["position"].copy())
    trace["tilts"].append(state["tilt"].copy())
    trace["original_indices"].append(int(original_index))


def timed_outcome(trace, reset_position):
    positions = np.asarray(trace["positions"])
    tilts = np.asarray(trace["tilts"])
    original = np.asarray(trace["original_indices"])
    baseline_rows = np.flatnonzero(original >= 9)
    baseline_index = int(baseline_rows[0]) if len(baseline_rows) else 0
    baseline = positions[baseline_index, 0] if len(positions) else reset_position[0]
    after = positions[baseline_index:, 0]
    lift = after[:, 2] - baseline[2]
    lateral = np.linalg.norm(after[:, :2] - baseline[:2], axis=1)
    reached = np.flatnonzero((lift >= .025) & (lateral >= .02))
    pick = bool(len(reached))
    first = baseline_index + int(reached[0]) if pick else None
    failure = bool(np.max(tilts[:, 1:]) >= 45.0)
    return {
        "pick_success": pick, "neighbor_failure": failure,
        "safe_completion": bool(pick and not failure),
        "time_to_pick_from_episode_start_s": (
            float((first + 1) * CONTROL_PERIOD_S) if first is not None else None),
        "time_to_pick_after_baseline_s": (
            float((first - baseline_index) * CONTROL_PERIOD_S) if first is not None else None),
        "executed_control_steps": len(positions),
        "episode_execution_time_s": float(len(positions) * CONTROL_PERIOD_S),
    }


def replay_arm(sim, episode, decisions, reset_seed):
    sim.reset(reset_seed)
    reset_position = sim.block_diagnostics()["position"].copy()
    by_start = {int(row["chunk_start"]): row for row in decisions}
    trace = {"positions": [], "tilts": [], "original_indices": []}
    i = 0
    while i < len(episode.actions):
        is_decision = i >= 2 and (i - 2) % HORIZON == 0 and i + HORIZON <= len(episode.actions)
        if not is_decision:
            sim.execute(episode.actions[i]); _record(trace, sim, i); i += 1
            continue
        row = by_start.pop(i)
        original = np.asarray(episode.actions[i:i + HORIZON], np.float32)
        if row["reobserved"]:
            sim.execute(hold_action(current_action(sim))); _record(trace, sim, i - 1)
        candidates = scaled_action_candidates(original, current_action(sim))
        scale_index = ACTION_SCALES.index(float(row["chosen_scale"]))
        for offset, action in enumerate(candidates[scale_index]):
            sim.execute(action); _record(trace, sim, i + offset)
        i += HORIZON
    if by_start:
        raise RuntimeError(f"unused decisions at chunks {sorted(by_start)}")
    return timed_outcome(trace, reset_position)


def _run_episode(job):
    row, lmdb, xml = job
    replay = JengaReplay(lmdb); episode = replay.episode(row["episode_id"]); replay.close()
    sim = DirectJengaSim(xml)
    try:
        timed = {arm: replay_arm(sim, episode, row["arms"][arm]["decisions"],
                                 RESET_BASE + int(row["episode_id"])) for arm in ARMS}
    finally:
        sim.close()
    for arm in ARMS:
        expected = row["arms"][arm]
        # The original run interleaved counterfactual simulation and state restoration. MuJoCo's
        # FULLPHYSICS snapshot omits solver warm-start, so an action-only replay can occasionally
        # straddle the 45-degree grading boundary. Pick timing must reproduce; immutable original
        # physical labels remain the grade, and every neighbor mismatch is reported explicitly.
        if timed[arm]["pick_success"] != expected["pick_success"]:
            raise RuntimeError(f"pick replay mismatch ep{row['episode_id']} {arm}")
        timed[arm]["replay_neighbor_failure"] = timed[arm]["neighbor_failure"]
        timed[arm]["neighbor_failure_replay_mismatch"] = bool(
            timed[arm]["neighbor_failure"] != expected["neighbor_failure"])
        timed[arm]["neighbor_failure"] = bool(expected["neighbor_failure"])
        timed[arm]["safe_completion"] = bool(expected["safe_completion"])
    return {"episode_id": row["episode_id"], "arms": timed}


def _distribution(values):
    values = np.asarray(values, float)
    return {"count": len(values), "mean_s": float(values.mean()),
            "median_s": float(np.median(values)),
            "p25_s": float(np.quantile(values, .25)),
            "p75_s": float(np.quantile(values, .75))}


def summarise(rows):
    summary = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        success = [row["time_to_pick_from_episode_start_s"] for row in values
                   if row["pick_success"]]
        safe = [row["time_to_pick_from_episode_start_s"] for row in values
                if row["safe_completion"]]
        summary[arm] = {
            "pick_time_successes": _distribution(success),
            "pick_time_safe_completions": _distribution(safe),
            "mean_episode_execution_time_s": float(np.mean([
                row["episode_execution_time_s"] for row in values])),
            "neighbor_failure_replay_mismatches": int(sum(
                row["neighbor_failure_replay_mismatch"] for row in values)),
        }
    baseline = {row["episode_id"]: row["arms"]["no_monitor"] for row in rows}
    for arm in ARMS[1:]:
        treatment = {row["episode_id"]: row["arms"][arm] for row in rows}
        common = [key for key in baseline if baseline[key]["pick_success"]
                  and treatment[key]["pick_success"]]
        deltas = [treatment[key]["time_to_pick_from_episode_start_s"]
                  - baseline[key]["time_to_pick_from_episode_start_s"] for key in common]
        summary[arm]["paired_time_to_pick_minus_no_monitor"] = _distribution(deltas)
        safe_common = [key for key in baseline if baseline[key]["safe_completion"]
                       and treatment[key]["safe_completion"]]
        safe_deltas = [treatment[key]["time_to_pick_from_episode_start_s"]
                       - baseline[key]["time_to_pick_from_episode_start_s"]
                       for key in safe_common]
        summary[arm]["paired_safe_time_to_pick_minus_no_monitor"] = _distribution(safe_deltas)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    verify()
    if sha256_file(RESULT) != EXPECTED_RESULT_SHA256:
        raise SystemExit("immutable intervention result hash mismatch")
    output = Path(args.output)
    if output.resolve() != OUTPUT.resolve() or output.exists():
        raise SystemExit(f"time result must be new canonical output {OUTPUT}")
    source = json.loads(RESULT.read_text())
    with tempfile.TemporaryDirectory(prefix="jenga_intervention_time_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        jobs = [(row, args.lmdb, xml) for row in source["rows"]]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_episode, job) for job in jobs]
            for number, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                print(f"timed replay {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: int(row["episode_id"]))
    result = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(RESULT.relative_to(ROOT)), "source_sha256": EXPECTED_RESULT_SHA256,
        "protocol": {"control_period_s": CONTROL_PERIOD_S,
                     "time_to_pick": "first post-baseline control step with middle lift >=2.5 cm "
                                     "and lateral displacement >=2 cm",
                     "policy": "replay immutable chosen scales/reobservations; no monitor rerun",
                     "outcome_grade": "immutable original result; action-only replay separately "
                                      "reports 45-degree neighbor mismatches caused by omitted "
                                      "MuJoCo solver warm-start state"},
        "summary": summarise(rows), "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
