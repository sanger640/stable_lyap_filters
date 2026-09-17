"""Ordered function-fitting replacement for local delta-z/HDBSCAN.

Real encoded MuJoCo responses are scored first.  Predicted responses are evaluated
only if the predeclared real quiet-control false-positive gate passes.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ordered_change_point import detect_ordered_jump, robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import read_probe_cache  # noqa: E402


def physical_boundary(outcomes, minimum=2):
    values = np.asarray(outcomes, bool)
    return bool(min(values.sum(), (~values).sum()) >= int(minimum))


def physical_split_scalars(scalars, outcomes):
    """Midpoints where ordered physical outcomes switch state."""
    s = np.asarray(scalars, np.float64)
    y = np.asarray(outcomes, bool)
    changed = np.flatnonzero(y[1:] != y[:-1])
    return ((s[changed] + s[changed + 1]) / 2.0).tolist()


def confusion(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = int((truth & predicted).sum()); fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum()); tn = int((~truth & ~predicted).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "false_positive_rate": fp / max(fp + tn, 1)}


def score_responses(scalars, responses, physical, metadata, *, dimension, degree,
                    min_side, min_jump_ratio, scale):
    rows = []
    for meta, response, outcome in zip(metadata, responses, physical):
        result = detect_ordered_jump(
            scalars, response[:, :dimension], degree=degree, min_side=min_side,
            min_jump_ratio=min_jump_ratio, scale=scale[:dimension])
        true_splits = physical_split_scalars(scalars, outcome)
        split_error = None
        if result.split_scalar is not None and true_splits:
            split_error = min(abs(result.split_scalar - value) for value in true_splits)
        row = dict(meta)
        row.update({
            "physical_boundary": physical_boundary(outcome),
            "physical_counts": {"intact": int((~outcome).sum()),
                                "toppled": int(outcome.sum())},
            "physical_split_scalars": true_splits,
            "alarm": result.alarm,
            "split_index": result.split_index,
            "split_scalar": result.split_scalar,
            "split_error": split_error,
            "jump_evidence_bic": result.jump_evidence_bic,
            "jump_norm": result.jump_norm,
            "adjacent_jump_ratio": result.adjacent_jump_ratio,
            "null_bic": result.null_bic,
            "continuous_bic": result.continuous_bic,
            "discontinuous_bic": result.discontinuous_bic,
            "coverage": result.coverage,
        })
        rows.append(row)
    return rows


def summarise(rows):
    truth = [row["physical_boundary"] for row in rows]
    alarms = [row["alarm"] for row in rows]
    quiet = [row for row in rows if row["stratum"] == "quiet_control"]
    errors = [row["split_error"] for row in rows
              if row["alarm"] and row["physical_boundary"] and row["split_error"] is not None]
    return {
        "confusion": confusion(truth, alarms),
        "quiet_control_alarms": int(sum(row["alarm"] for row in quiet)),
        "quiet_controls": len(quiet),
        "quiet_control_false_positive_rate": (
            sum(row["alarm"] for row in quiet) / max(len(quiet), 1)),
        "median_true_positive_split_error": float(np.median(errors)) if errors else None,
        "mean_coverage": float(np.mean([row["coverage"] for row in rows])),
    }


def bootstrap_confidence(rows, seed=0, samples=2000):
    """Episode-level bootstrap intervals for precision, recall and FPR."""
    episodes = sorted(set(row["episode_id"] for row in rows), key=int)
    grouped = {episode: [row for row in rows if row["episode_id"] == episode]
               for episode in episodes}
    rng = np.random.default_rng(seed); values = []
    for _ in range(int(samples)):
        selected = rng.choice(episodes, len(episodes), replace=True)
        sample = [row for episode in selected for row in grouped[episode]]
        metric = confusion([row["physical_boundary"] for row in sample],
                           [row["alarm"] for row in sample])
        values.append([metric["precision"], metric["recall"],
                       metric["false_positive_rate"]])
    values = np.asarray(values)
    return {name: [float(x) for x in np.quantile(values[:, i], [0.025, 0.975])]
            for i, name in enumerate(("precision", "recall", "false_positive_rate"))}


def diagnostic_plot(path, scalars, actual, rows, dimension):
    import matplotlib.pyplot as plt
    categories = [
        ("true positive", lambda r: r["alarm"] and r["physical_boundary"]),
        ("false negative", lambda r: not r["alarm"] and r["physical_boundary"]),
        ("quiet true negative", lambda r: not r["alarm"] and r["stratum"] == "quiet_control"),
        ("quiet false positive", lambda r: r["alarm"] and r["stratum"] == "quiet_control"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (title, predicate) in zip(axes.flat, categories):
        indices = [i for i, row in enumerate(rows) if predicate(row)]
        if not indices:
            axis.text(0.5, 0.5, "none", ha="center", va="center")
            axis.set_title(title); continue
        i = indices[0]; row = rows[i]
        response = actual[i, :, :dimension]
        response = response - response[0]
        axis.plot(scalars, np.linalg.norm(response, axis=1), "o-", ms=3)
        if row["split_scalar"] is not None:
            axis.axvline(row["split_scalar"], color="tab:red", linestyle="--",
                         label="fitted split")
        for j, split in enumerate(row["physical_split_scalars"]):
            axis.axvline(split, color="black", alpha=0.5, linestyle=":",
                         label="physical switch" if j == 0 else None)
        axis.set_title(f"{title}: ep{row['episode_id']} chunk {row['chunk_start']}")
        axis.set_xlabel("probe strength s"); axis.set_ylabel("||delta z(s)-delta z(min)||")
        axis.legend(fontsize=8)
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(
        ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/ordered_change_points.json"))
    ap.add_argument("--plot", default=str(
        ROOT / "results/jenga/ordered_change_points.png"))
    ap.add_argument("--dimension", type=int, default=4)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--min-side", type=int, default=8)
    ap.add_argument("--min-jump-ratio", type=float, default=3.0)
    ap.add_argument("--real-control-fpr-gate", type=float, default=0.05)
    ap.add_argument("--external-gate", help="Optional held-out result whose real gate must also pass")
    args = ap.parse_args()

    metadata, scalars, predicted, actual, physical = read_probe_cache(args.cache)
    scale = robust_component_scale(actual)
    sensitivity = {}
    actual_by_dimension = {}
    for dimension in (2, 4, 8, 16):
        rows = score_responses(
            scalars, actual, physical, metadata, dimension=dimension, degree=args.degree,
            min_side=args.min_side, min_jump_ratio=args.min_jump_ratio, scale=scale)
        actual_by_dimension[dimension] = rows
        sensitivity[str(dimension)] = summarise(rows)

    actual_rows = actual_by_dimension[args.dimension]
    actual_summary = summarise(actual_rows)
    actual_summary["bootstrap_95_percent"] = bootstrap_confidence(actual_rows)
    internal_gate_passed = (
        actual_summary["quiet_control_false_positive_rate"] <= args.real_control_fpr_gate)
    external_gate_passed = None
    if args.external_gate:
        external = json.loads(Path(args.external_gate).read_text())
        external_gate_passed = bool(external["real_control_gate"]["passed"])
    gate_passed = internal_gate_passed and external_gate_passed is not False
    predicted_rows = None; predicted_summary = None
    exploratory_predicted_summary = None
    if internal_gate_passed:
        predicted_rows = score_responses(
            scalars, predicted, physical, metadata, dimension=args.dimension,
            degree=args.degree, min_side=args.min_side,
            min_jump_ratio=args.min_jump_ratio, scale=scale)
        predicted_summary = summarise(predicted_rows)
        predicted_summary["bootstrap_95_percent"] = bootstrap_confidence(predicted_rows)
        if not gate_passed:
            exploratory_predicted_summary = predicted_summary
            predicted_summary = None

    result = {
        "protocol": {"chunks": len(metadata), "probes": len(scalars), "horizon": 8,
                     "held_tail": 5, "dimension": args.dimension, "degree": args.degree,
                     "min_side": args.min_side, "min_jump_ratio": args.min_jump_ratio},
        "detector": "BIC continuous piecewise polynomial versus discontinuous counterpart; observed adjacent jump ratio gate",
        "physical_boundary_chunks": int(sum(physical_boundary(row) for row in physical)),
        "real_control_gate": {"maximum_fpr": args.real_control_fpr_gate,
                              "internal_passed": internal_gate_passed,
                              "external_gate": args.external_gate,
                              "external_passed": external_gate_passed,
                              "passed": gate_passed},
        "actual_latent": actual_summary,
        "actual_dimension_sensitivity": sensitivity,
        "predicted_latent": predicted_summary,
        "exploratory_predicted_latent": exploratory_predicted_summary,
        "prediction_evaluation_withheld": not gate_passed,
        "actual_rows": actual_rows,
        "predicted_rows": predicted_rows,
        "note": "Prediction is scored only after the encoded-real quiet-control gate passes.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    diagnostic_plot(args.plot, scalars, actual, actual_rows, args.dimension)
    print("ordered change-point result")
    print("actual", actual_summary)
    print("dimension sensitivity", {key: value["quiet_control_false_positive_rate"]
                                    for key, value in sensitivity.items()})
    print("real control gate", "PASS" if gate_passed else "FAIL")
    print("predicted", predicted_summary if gate_passed else "WITHHELD")
    if exploratory_predicted_summary is not None:
        print("exploratory prediction observed before external gate correction",
              exploratory_predicted_summary)
    print(f"wrote {output} and {args.plot}")


if __name__ == "__main__":
    main()
