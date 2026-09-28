"""Matched 100-episode intervention test for contact-regime U1."""
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

import contact_regime_u1 as u1
from intervention_policy import ACTION_SCALES, choose_oracle_candidate, choose_robust_candidate
from jenga_regime_monitor_v0_freeze import sha256_file
from systems.contact_benchmarks import InsertionBenchmark, PushingBenchmark

OUT = u1.OUT / "intervention"
MANIFEST = OUT / "manifest.json"
RESULT = OUT / "ground_truth_100.json"
EPISODES = 100
ARMS = ("no_monitor", "reobserve", "wrapper", "oracle")
PROTOCOL = {
    "id": "contact-intervention-u1",
    "tasks": ["pushing", "insertion"],
    "episodes_per_task": EPISODES,
    "matched_arms": list(ARMS),
    "detector": "frozen contact-regime U1 application of frozen Regime Monitor v0",
    "decision_points": "non-overlapping H=8 chunks over a 96-step task trajectory",
    "reobserve": "one 0.1-second hold step followed by one v0 rerun",
    "candidates": list(ACTION_SCALES),
    "candidate_definition": "contract the intended task-space control chunk toward current control",
    "wrapper_choice": "fewest v0 consequential pairs, ties closest to intended action",
    "oracle_choice": "avoid local task failure over H8+hold30, ties closest to intended action",
    "episode_grading": "pushing goal/support and insertion depth/jam; labels never enter wrapper",
}
FROZEN_FILES = ("eval/contact_intervention_u1.py", "src/intervention_policy.py",
                "results/contact_benchmarks/u1/manifest.json",
                "results/jenga/regime_monitor_v0/manifest.json")


def _digest(protocol, files):
    encoded = json.dumps({"protocol": protocol, "files_sha256": files},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze():
    u1.verify()
    if MANIFEST.exists(): raise SystemExit("intervention U1 already frozen")
    files = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files, "protocol_sha256": _digest(PROTOCOL, files)}
    OUT.mkdir(parents=True, exist_ok=True); MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    return value


def verify():
    u1.verify(); value = json.loads(MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL or tuple(value.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("intervention U1 protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256_file(ROOT / name) != expected: raise SystemExit(f"intervention U1 changed: {name}")
    if _digest(PROTOCOL, value["files_sha256"]) != value["protocol_sha256"]:
        raise SystemExit("intervention U1 digest mismatch")
    return value


def episode_spec(task, episode):
    rng = np.random.default_rng(190000 + episode + (0 if task == "pushing" else 10000))
    phase = np.linspace(0., 1., 96)
    if task == "pushing":
        object_y = float(rng.uniform(-.05, .05)); target_y = float(object_y + rng.uniform(-.012, .012))
        end_x = float(rng.uniform(.54, .78))
        controls = np.column_stack([phase * end_x, phase * target_y])
        return {"object_y": object_y, "target_y": target_y, "end_x": end_x}, controls
    lateral = rng.uniform(-.014, .014, 2); yaw = float(rng.uniform(-.12, .12))
    z_end = float(rng.uniform(-.155, -.135))
    controls = np.column_stack([phase*lateral[0], phase*lateral[1], phase*z_end, phase*yaw])
    return {"lateral_x": float(lateral[0]), "lateral_y": float(lateral[1]),
            "z_end": z_end, "yaw": yaw}, controls


def _current(env): return np.asarray(env.data.ctrl, float).copy()


def _candidates(chunk, current):
    return np.stack([current + scale * (np.asarray(chunk) - current) for scale in ACTION_SCALES])


def monitor(env, task, snapshot, chunk, snippets):
    traces = np.stack([u1._rollout(env, task, snapshot, chunk, noise)[0]
                       for noise in u1.map_noise(task, snippets)])
    row = {"task": task, "episode": -1, "chunk_start": -1, "split": "runtime",
           "cohort": "unlabelled", "failure_count": -1, "success_count": -1,
           "mode_counts": {}}
    result = u1._evaluate_state((0, row, snapshot, chunk, traces, snippets))
    env.restore(snapshot)
    return {"alarm": bool(result["alarm"]),
            "consequential_pairs": int(result.get("consequential_pairs", 0)),
            "initial_alarm": bool(result["initial_alarm"]),
            "boundary_alarm": bool(result["boundary_refined_alarm"])}


def physical_grade(env, task, snapshot, chunk):
    env.restore(snapshot)
    for command in chunk: env.step(command)
    for _ in range(u1.HOLD): env.step(chunk[-1])
    outcome = env.outcome()
    # The second element is a deterministic generic tie-break. Deviation remains the first tie-break
    # in choose_oracle_candidate, so this value matters only among equally scaled candidates.
    severity = (-outcome.values["object_position_m"][2] if task == "pushing"
                else outcome.values["peg_position_m"][2])
    env.restore(snapshot)
    return bool(outcome.failure), float(severity)


def run_arm(task, episode, arm, snippets):
    env = PushingBenchmark(width=64, height=64) if task == "pushing" else InsertionBenchmark(width=64, height=64)
    try:
        params, controls = episode_spec(task, episode); u1._reset(env, task, params)
        decisions = []
        for start in range(0, len(controls), u1.HORIZON):
            original = controls[start:start + u1.HORIZON]; chosen = original
            decision = {"chunk_start": start, "initial_alarm": False, "persistent_alarm": False,
                        "reobserved": False, "chosen_scale": 1.0}
            if arm != "no_monitor":
                snapshot = env.snapshot(); first = monitor(env, task, snapshot, original, snippets)
                decision["initial_alarm"] = first["alarm"]
                decision["initial_consequential_pairs"] = first["consequential_pairs"]
                if first["alarm"]:
                    env.step(_current(env)); decision["reobserved"] = True
                    after = env.snapshot(); second = monitor(env, task, after, original, snippets)
                    decision["persistent_alarm"] = second["alarm"]
                    decision["persistent_consequential_pairs"] = second["consequential_pairs"]
                    if second["alarm"] and arm in ("wrapper", "oracle"):
                        candidates = _candidates(original, _current(env))
                        if arm == "wrapper":
                            evidence = [second] + [monitor(env, task, after, c, snippets)
                                                   for c in candidates[1:]]
                            index = choose_robust_candidate(evidence).index
                            decision["candidate_consequential_pairs"] = [
                                e["consequential_pairs"] for e in evidence]
                        else:
                            grades = [physical_grade(env, task, after, c) for c in candidates]
                            index = choose_oracle_candidate(grades)
                            decision["candidate_failures"] = [g[0] for g in grades]
                        chosen = candidates[index]; decision["chosen_scale"] = ACTION_SCALES[index]
                    env.restore(after)
            for command in chosen: env.step(command)
            decisions.append(decision)
        outcome = env.outcome()
        return {"success": bool(outcome.success), "failure": bool(outcome.failure),
                "safe_completion": bool(outcome.success and not outcome.failure), "mode": outcome.mode,
                "decision_points": len(decisions),
                "initial_alarms": sum(d["initial_alarm"] for d in decisions),
                "persistent_alarms": sum(d["persistent_alarm"] for d in decisions),
                "reobservations": sum(d["reobserved"] for d in decisions),
                "modified_chunks": sum(d["chosen_scale"] < 1 for d in decisions),
                "added_control_seconds": .1 * sum(d["reobserved"] for d in decisions),
                "decisions": decisions}
    finally: env.close()


def _run_episode(job):
    task, episode, snippets = job
    return {"task": task, "episode": episode,
            "arms": {arm: run_arm(task, episode, arm, snippets) for arm in ARMS}}


def summarise(rows):
    out = {}
    for task in ("pushing", "insertion"):
        out[task] = {}
        selected = [r for r in rows if r["task"] == task]
        for arm in ARMS:
            values = [r["arms"][arm] for r in selected]
            out[task][arm] = {"episodes": len(values),
                              "successes": sum(v["success"] for v in values),
                              "failures": sum(v["failure"] for v in values),
                              "safe_completions": sum(v["safe_completion"] for v in values),
                              "episodes_reobserved": sum(v["reobservations"] > 0 for v in values),
                              "reobservations": sum(v["reobservations"] for v in values),
                              "persistent_alarms": sum(v["persistent_alarms"] for v in values),
                              "modified_chunks": sum(v["modified_chunks"] for v in values),
                              "added_control_seconds": sum(v["added_control_seconds"] for v in values)}
    return out


def run(workers):
    manifest = verify()
    if RESULT.exists(): raise SystemExit("canonical intervention U1 result exists")
    snippets = np.load(ROOT / "results/jenga/holdout_stage0_cache.npz")["snippets"]
    jobs = [(task, episode, snippets) for task in ("pushing", "insertion") for episode in range(EPISODES)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_episode, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result()); print(f"episode {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["task"], r["episode"]))
    value = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": summarise(rows), "rows": rows}
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("verify")
    runner = sub.add_parser("run"); runner.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze": print(freeze()["protocol_sha256"])
    elif args.command == "verify": print(verify()["protocol_sha256"])
    else: run(args.workers)


if __name__ == "__main__": main()
