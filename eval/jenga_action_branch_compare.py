"""Aggregate the frozen action-conditioned D2 monitor across seeds."""
import glob
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "w6_cw_d2_s"


def main():
    paths = glob.glob(str(ROOT / f"results/jenga/bench_eval/{PREFIX}*_action_branch.json"))
    runs = []
    for path in paths:
        match = re.fullmatch(PREFIX + r"(\d+)_action_branch\.json", Path(path).name)
        if match:
            runs.append((int(match.group(1)), json.loads(Path(path).read_text())))
    runs.sort(key=lambda item: item[0])
    if len({seed for seed, _ in runs}) != len(runs):
        raise SystemExit("duplicate seeds")
    per_seed, misses, quiet_alarms = [], {}, {}
    evidence = {"fork": [], "quiet": []}
    for seed, run in runs:
        rows = [r for r in run["rows"] if r["split"] == "test" and r["scale"] == "1.0"]
        forks = [r for r in rows if r["class"] == "topple_fork"]
        quiet = [r for r in rows if r["class"] == "quiet"]
        per_seed.append({"seed": seed, "recall": float(np.mean([r["alarm"] for r in forks])),
                         "quiet_alarm_rate": float(np.mean([r["alarm"] for r in quiet]))})
        for row in forks:
            key = f"{row['episode_id']}:{row['chunk_start']}"
            misses[key] = misses.get(key, 0) + int(not row["alarm"])
            evidence["fork"].append(min(row["early_delta_description_length"],
                                         row["full_delta_description_length"]))
        for row in quiet:
            key = f"{row['episode_id']}:{row['chunk_start']}"
            quiet_alarms[key] = quiet_alarms.get(key, 0) + int(row["alarm"])
            evidence["quiet"].append(min(row["early_delta_description_length"],
                                          row["full_delta_description_length"]))
    n = len(runs)
    summary = {
        "seeds": [seed for seed, _ in runs],
        "recall": {"mean": float(np.mean([r["recall"] for r in per_seed])),
                   "min": float(np.min([r["recall"] for r in per_seed])),
                   "max": float(np.max([r["recall"] for r in per_seed]))},
        "quiet_alarm_rate": {
            "mean": float(np.mean([r["quiet_alarm_rate"] for r in per_seed])),
            "min": float(np.min([r["quiet_alarm_rate"] for r in per_seed])),
            "max": float(np.max([r["quiet_alarm_rate"] for r in per_seed]))},
        "robust_blind_count": sum(value / n >= 0.8 for value in misses.values()),
        "robust_blind_forks": sorted(key for key, value in misses.items() if value / n >= 0.8),
        "robust_quiet_alarm_count": sum(value / n >= 0.8 for value in quiet_alarms.values()),
        "robust_quiet_alarms": sorted(key for key, value in quiet_alarms.items()
                                      if value / n >= 0.8),
        "evidence": {name: {"median": float(np.median(values)),
                            "p90": float(np.quantile(values, .9))}
                     for name, values in evidence.items()},
        "per_seed": per_seed,
    }
    ground_path = ROOT / "results/jenga/action_branch_ground_truth.json"
    if ground_path.exists():
        ground = json.loads(ground_path.read_text())["summary"]["test"]["1.0"]
        ground_recall = ground["topple_fork"]["rate"]
        ground_quiet = ground["quiet"]["rate"]
        summary["ground_truth"] = {
            "recall": ground_recall,
            "quiet_alarm_rate": ground_quiet,
        }
        summary["d2_minus_ground_truth"] = {
            "recall": summary["recall"]["mean"] - ground_recall,
            "quiet_alarm_rate": summary["quiet_alarm_rate"]["mean"] - ground_quiet,
        }
    result = {"protocol": {"runtime_calibration": "none", "model": "D2",
                           "scale": 1.0, "robust": "miss/alarm in >=80% of seeds",
                           "matched_control": "simulator and D2 use identical probes, monitor, "
                                              "block-pose features and hold 5,10,20,29,30 samples"},
              "summary": summary}
    output = ROOT / "results/jenga/action_branch_d2.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
