"""Run the calibration-free action-branch monitor on cached simulator trajectories.

This is the control experiment for D2.  It uses the same frozen states, action perturbations,
five trajectory times, feature transform, and monitor decision; only the trajectory source is the
simulator rather than the learned world model.  Physical classes are joined after inference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import result_dict, smooth_vs_branch_alarm  # noqa: E402
from jenga_action_branch import action_errors, pose_features, summarise  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, SPLITS, verify  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT  # noqa: E402


def evaluate(selected_scales):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    rows = []
    early_stop = RECORD_AT.index(10) + 1
    for split, spec in SPLITS.items():
        metadata = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
        index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(metadata)}
        cache = np.load(ROOT / spec["cache"], allow_pickle=False)
        for i, episode_id in enumerate(bench[f"{split}_episode"]):
            chunk = int(bench[f"{split}_chunk"][i])
            cache_row = index[(str(episode_id), chunk)]
            for scale_index, value in enumerate(EVAL_CONFIG["scales"]):
                if value not in selected_scales:
                    continue
                windows = bench[f"{split}_windows"][i, scale_index]
                trajectory = pose_features(cache["pose"][cache_row, scale_index])
                result = smooth_vs_branch_alarm(
                    action_errors(windows), trajectory[:, :early_stop], trajectory)
                rows.append({
                    "split": split,
                    "episode_id": str(episode_id),
                    "chunk_start": chunk,
                    "scale": str(value),
                    "class": str(bench[f"{split}_classes"][i, scale_index]),
                    "topple_count": int(bench[f"{split}_topples"][i, scale_index]),
                    **result_dict(result),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/action_branch_ground_truth.json"))
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0],
                    choices=EVAL_CONFIG["scales"])
    args = ap.parse_args()
    rows = evaluate(args.scales)
    result = {
        "protocol": {
            "calibration_free": True,
            "trajectory_source": "cached simulator ground-truth block poses",
            "input": "H=8 xyz execution perturbations and block pose at hold 5,10,20,29,30",
            "comparison": "identical probes, temporal samples, pose features, and frozen monitor "
                          "used by eval/jenga_action_branch.py",
            "decision": "penalised branch evidence is positive at both hold 10 and hold 30",
            "labels": "physical classes are grading only",
            "uses_reference_states": False,
            "scales": args.scales,
        },
        "summary": summarise(rows, args.scales),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["test"]["1.0"], indent=2))


if __name__ == "__main__":
    main()
