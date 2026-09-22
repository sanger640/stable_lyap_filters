"""Aggregate calibration-free monitor results across independently trained seeds."""
import argparse
import glob
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RULES = ("pc1_persistent_alarm", "multi_exact_alarm", "multi_coarse_alarm",
         "multi_dominant_alarm")


def seed(path, prefix):
    match = re.fullmatch(re.escape(prefix) + r"(\d+)_calfree\.json", Path(path).name)
    if not match:
        raise ValueError(f"unexpected result name: {path}")
    return int(match.group(1))


def aggregate(paths, prefix):
    runs = [(seed(path, prefix), json.loads(Path(path).read_text())) for path in paths]
    runs.sort(key=lambda item: item[0])
    if len({s for s, _ in runs}) != len(runs):
        raise ValueError("duplicate seeds")
    result = {"seeds": [s for s, _ in runs], "rules": {}}
    for rule in RULES:
        rates, missed, quiet_alarm = [], {}, {}
        for _, run in runs:
            rows = [r for r in run["rows"] if r["split"] == "test" and r["scale"] == "1.0"]
            forks = [r for r in rows if r["class"] == "topple_fork"]
            quiet = [r for r in rows if r["class"] == "quiet"]
            rates.append({"recall": float(np.mean([r[rule] for r in forks])),
                          "quiet_alarm_rate": float(np.mean([r[rule] for r in quiet]))})
            for row in forks:
                key = f"{row['episode_id']}:{row['chunk_start']}"
                missed[key] = missed.get(key, 0) + int(not row[rule])
            for row in quiet:
                key = f"{row['episode_id']}:{row['chunk_start']}"
                quiet_alarm[key] = quiet_alarm.get(key, 0) + int(row[rule])
        n = len(runs)
        result["rules"][rule] = {
            "recall": {"mean": float(np.mean([r["recall"] for r in rates])),
                       "min": float(np.min([r["recall"] for r in rates])),
                       "max": float(np.max([r["recall"] for r in rates]))},
            "quiet_alarm_rate": {
                "mean": float(np.mean([r["quiet_alarm_rate"] for r in rates])),
                "min": float(np.min([r["quiet_alarm_rate"] for r in rates])),
                "max": float(np.max([r["quiet_alarm_rate"] for r in rates]))},
            "robust_blind_count": sum(v / n >= 0.8 for v in missed.values()),
            "robust_blind_forks": sorted(k for k, v in missed.items() if v / n >= 0.8),
            "robust_quiet_alarm_count": sum(v / n >= 0.8 for v in quiet_alarm.values()),
            "per_seed": rates,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="w6_cw_d2_s")
    ap.add_argument("--output", default=str(ROOT / "results/jenga/calibration_free_d2.json"))
    args = ap.parse_args()
    paths = glob.glob(str(ROOT / f"results/jenga/bench_eval/{args.prefix}*_calfree.json"))
    if not paths:
        raise SystemExit(f"no results for prefix {args.prefix}")
    result = {"protocol": {"runtime_calibration": "none", "failure_labels": "grading only",
                           "scale": 1.0, "robust": "miss/alarm in >=80% of seeds"},
              **aggregate(paths, args.prefix)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["rules"], indent=2))


if __name__ == "__main__":
    main()
