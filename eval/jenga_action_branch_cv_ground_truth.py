"""Out-of-sample action-branch test on the frozen ground-truth Jenga trajectories.

Each of eight folds hides eight of the 64 probe trajectories. Candidate branch discovery and GP
fitting use the other 56; hidden branch membership is inferred from action location alone. The
alarm fires only when branch models predict unseen trajectories better than one smooth model at
both hold 10 and hold 30. Physical classes are joined after inference for grading only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (cross_validated_smooth_vs_branch_alarm,  # noqa: E402
                                   result_dict)
from jenga_action_branch import action_errors, pose_features, summarise  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, SPLITS, verify  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT  # noqa: E402


def _score(job):
    actions, early, full = job
    return result_dict(cross_validated_smooth_vs_branch_alarm(actions, early, full))


def evaluate(selected_splits, workers):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    early_stop = RECORD_AT.index(10) + 1
    jobs, metadata = [], []
    for split in selected_splits:
        spec = SPLITS[split]
        stage0 = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
        index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0)}
        cache = np.load(ROOT / spec["cache"], allow_pickle=False)
        scale_index = EVAL_CONFIG["scales"].index(1.0)
        for i, episode_id in enumerate(bench[f"{split}_episode"]):
            chunk = int(bench[f"{split}_chunk"][i])
            cache_row = index[(str(episode_id), chunk)]
            windows = bench[f"{split}_windows"][i, scale_index]
            trajectory = pose_features(cache["pose"][cache_row, scale_index])
            jobs.append((action_errors(windows), trajectory[:, :early_stop], trajectory))
            metadata.append({
                "split": split,
                "episode_id": str(episode_id),
                "chunk_start": chunk,
                "scale": "1.0",
                "class": str(bench[f"{split}_classes"][i, scale_index]),
                "topple_count": int(bench[f"{split}_topples"][i, scale_index]),
            })
    if workers == 1:
        scores = map(_score, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        scores = pool.map(_score, jobs, chunksize=1)
    try:
        rows = [{**meta, **score} for meta, score in zip(metadata, scores)]
    finally:
        if workers != 1:
            pool.shutdown()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/action_branch_cv_ground_truth.json"))
    args = ap.parse_args()
    rows = evaluate(args.splits, args.workers)
    result = {
        "protocol": {
            "calibration_free": True,
            "trajectory_source": "cached simulator ground-truth block poses",
            "input": "64 H=8 xyz perturbations; block pose at hold 5,10,20,29,30",
            "cross_validation": "8 deterministic action-balanced folds; 8 trajectories hidden "
                                "per fold; hidden branch assigned by 5 action-nearest visible probes",
            "decision": "total held-out branch prediction squared error is below the one-smooth "
                        "error at both hold 10 and hold 30",
            "labels": "physical classes are grading only",
            "uses_reference_states": False,
            "splits": args.splits,
        },
        "summary": summarise(rows, [1.0]),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for split in args.splits:
        print(split, json.dumps(result["summary"][split]["1.0"], indent=2))


if __name__ == "__main__":
    main()
