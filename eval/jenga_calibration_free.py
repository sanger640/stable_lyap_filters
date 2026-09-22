"""Calibration-free fork monitor on the frozen Jenga benchmark.

The dynamics checkpoint may be trained with D2 intervention consistency, but the runtime alarm
uses no development examples, quiet-score percentile, failure label, task label, or fitted distance
threshold.  For the 64 execution-noise counterfactuals it asks whether predicted endings contain
at least two BIC-supported, Ashman-separated modes at hold steps 10 and 30, with at least two
probes per mode, and whether the dominant binary partition persists (at most one probe changes
side).  BIC supplies its own complexity penalty; Ashman's D is dimensionless; the minority rule is
fixed by the probe budget.  Physical classes are read only after alarms are frozen, for grading.

This is a sibling evaluator, not a modification of ``jenga_bench.py``: the original benchmark and
its manifest remain byte-frozen for historical comparisons.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from counterfactual_monitor import persistent_mode_alarm  # noqa: E402
from outcome_modes import coarse_persistent_fork, groupings_persist, multi_mode_test  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, checkpoint_identity, verify  # noqa: E402
from jenga_w5_eval import hold_index, load_model, predict  # noqa: E402

HOLDS = (10, 30)
RULE = (
    "original frozen PC1 two-mode rule: best split on PC1 at hold steps 10 and 30; two hard "
    "Gaussians beat one by BIC; Ashman D>2; each side has at least two of 64 probes; the binary "
    "partition matches with at most one probe misplaced"
)


def calibration_free_test(trajectories, model):
    """Return the frozen label-free alarm for one set of counterfactual trajectories."""
    endings_by_hold, multi = [], []
    for held in HOLDS:
        endings = 1000.0 * trajectories[:, hold_index(model, held), 0:9]
        endings_by_hold.append(endings)
        multi.append(multi_mode_test(endings))
    primary = persistent_mode_alarm(*endings_by_hold)
    early, late = multi
    both = early["groups"] >= 2 and late["groups"] >= 2
    exact = both and groupings_persist(early["labels"], late["labels"])
    coarse = both and coarse_persistent_fork(early["labels"], late["labels"])
    dominant = both and groupings_persist(early["dominant"], late["dominant"])
    return {
        "alarm": primary.alarm,
        "pc1_persistent_alarm": primary.alarm,
        "multi_exact_alarm": bool(exact),
        "multi_coarse_alarm": bool(coarse),
        "multi_dominant_alarm": bool(dominant),
        "groups_hold10": int(early["groups"]),
        "groups_hold30": int(late["groups"]),
        "sizes_hold10": early["sizes"],
        "sizes_hold30": late["sizes"],
        "persistent": primary.persistent_partition,
    }


def score_model(path, selected_scales):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(path, device)
    rows = []
    for split in ("dev", "test"):
        starts = bench[f"{split}_start"]
        windows = bench[f"{split}_windows"]
        for i, start in enumerate(starts):
            for k, value in enumerate(EVAL_CONFIG["scales"]):
                if value not in selected_scales:
                    continue
                trajectories = predict(model, scale, start, windows[i, k], device)
                test = calibration_free_test(trajectories, model)
                rows.append({
                    "split": split,
                    "episode_id": str(bench[f"{split}_episode"][i]),
                    "chunk_start": int(bench[f"{split}_chunk"][i]),
                    "scale": str(value),
                    "class": str(bench[f"{split}_classes"][i, k]),
                    "topple_count": int(bench[f"{split}_topples"][i, k]),
                    **test,
                })
    return rows


def rate(rows, cls):
    selected = [r for r in rows if r["class"] == cls]
    return {"states": len(selected), "alarms": sum(r["alarm"] for r in selected),
            "rate": float(np.mean([r["alarm"] for r in selected])) if selected else None}


def diagnostic_rates(rows, cls):
    selected = [r for r in rows if r["class"] == cls]
    keys = ("pc1_persistent_alarm", "multi_exact_alarm", "multi_coarse_alarm",
            "multi_dominant_alarm")
    return {key: {"alarms": sum(r[key] for r in selected),
                  "rate": float(np.mean([r[key] for r in selected])) if selected else None}
            for key in keys}


def summarise(rows, selected_scales):
    out = {}
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    for split in ("dev", "test"):
        out[split] = {}
        for scale in map(str, selected_scales):
            part = [r for r in rows if r["split"] == split and r["scale"] == scale]
            out[split][scale] = {
                cls: {**rate(part, cls), "diagnostics": diagnostic_rates(part, cls)}
                for cls in classes}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0],
                    choices=EVAL_CONFIG["scales"],
                    help="execution-noise scales to evaluate; 1x is the operational scale")
    args = ap.parse_args()
    rows = score_model(args.model, args.scales)
    result = {
        "protocol": {
            "calibration_free": True,
            "probes": EVAL_CONFIG["probes"],
            "execution_noise_scales": args.scales,
            "holds": list(HOLDS),
            "rule": RULE,
            "uses_development_scores_to_set_alarm": False,
            "uses_quiet_or_failure_labels_to_set_alarm": False,
            "physical_classes": "evaluation only, read after the alarm",
        },
        "model": checkpoint_identity(args.model),
        "summary": summarise(rows, args.scales),
        "rows": rows,
    }
    output = (Path(args.output) if args.output else ROOT / "results/jenga/bench_eval" /
              f"{Path(args.model).stem}_calfree.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["test"]["1.0"], indent=2))


if __name__ == "__main__":
    main()
