"""Read-only D18 robustness audit for the physical commitment delta-BIC target."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from consequence_monitor import commitment_delta_bic, whole_trajectory_separation_curves
from jenga_d6_set_response import sha256
from jenga_d12_model_ladder import load_records, response_normalizers


PROTOCOL = ROOT / "results/jenga/d18_commitment_robustness_protocol.json"
REPORT = ROOT / "results/jenga/d18_commitment_robustness_result.json"
HORIZON = 8
IMMEDIATE_HOLD = 5


def interpolate_missing(curve, missing):
    curve = np.asarray(curve, np.float64); keep = np.ones(len(curve), bool)
    keep[np.asarray(missing, int)] = False
    time = np.arange(len(curve), dtype=float)
    return np.interp(time, time[keep], curve[keep])


def robustness_curves(curve):
    """Return the nine frozen non-baseline temporal variants."""
    curve = np.asarray(curve, np.float64)
    if curve.shape != (39,):
        raise ValueError("D18 expects one H8 + hold30 curve with 39 samples")
    variants = {f"prefix_hold{hold}": curve[:HORIZON + 1 + hold]
                for hold in (15, 20, 25)}
    late = np.arange(HORIZON + IMMEDIATE_HOLD + 1, len(curve))  # indices 14..38
    for phase, name in ((0, "half_rate_even"), (1, "half_rate_odd")):
        sampled = late[(late - late[0]) % 2 == phase]
        keep = np.r_[np.arange(late[0]), sampled]
        if keep[-1] != len(curve) - 1:
            keep = np.r_[keep, len(curve) - 1]
        missing = np.setdiff1d(np.arange(len(curve)), keep)
        variants[name] = interpolate_missing(curve, missing)
    for start in (14, 19, 24, 29):
        variants[f"dropout_{start}_{start + 2}"] = interpolate_missing(
            curve, np.arange(start, start + 3))
    if len(variants) != 9:
        raise RuntimeError("D18 variant count changed")
    return variants


def evidence_for_group(actions, traces):
    # Match D13 exactly: boundary gaps use the fit response scale, but commitment consumes the
    # raw physical trace and performs its own equal-family normalization internally.
    curve = whole_trajectory_separation_curves(traces[-1:, :, :, :45])[0]
    action_difference = actions[-1, 0, :HORIZON] - actions[-1, 1, :HORIZON]
    baseline = commitment_delta_bic(curve, action_difference, HORIZON, IMMEDIATE_HOLD)[0]
    variants = {name: commitment_delta_bic(values, action_difference, HORIZON,
                                            IMMEDIATE_HOLD)[0]
                for name, values in robustness_curves(curve).items()}
    return float(baseline), variants


def evidence_for_fold(actions, traces):
    rows = [evidence_for_group(control, trajectory)
            for control, trajectory in zip(actions, traces)]
    names = tuple(rows[0][1])
    baseline = np.asarray([row[0] for row in rows], np.float64)
    variants = np.asarray([[row[1][name] for name in names] for row in rows], np.float64)
    return baseline, variants, names


def percentile_against_fit(values, fit_values):
    fit = np.sort(np.asarray(fit_values, np.float64))
    return np.searchsorted(fit, np.asarray(values), side="right") / len(fit)


def subset_summary(mask, baseline, variants, margin_percentile):
    mask = np.asarray(mask, bool); signs = variants > 0; baseline_sign = baseline > 0
    agreement = np.mean(signs == baseline_sign[:, None], axis=1)
    stable = np.all(signs == baseline_sign[:, None], axis=1)
    return {"count": int(mask.sum()),
            "sign_stable_fraction": float(np.mean(stable[mask])) if mask.any() else None,
            "sign_unstable_fraction": float(np.mean(~stable[mask])) if mask.any() else None,
            "variant_agreement_fraction_median": float(np.median(agreement[mask])) if mask.any() else None,
            "absolute_baseline_delta_bic_median": float(np.median(np.abs(baseline[mask]))) if mask.any() else None,
            "fit_margin_percentile_median": float(np.median(margin_percentile[mask])) if mask.any() else None,
            "minimum_variant_delta_bic_median": float(np.median(np.min(variants[mask], axis=1))) if mask.any() else None,
            "maximum_variant_delta_bic_median": float(np.median(np.max(variants[mask], axis=1))) if mask.any() else None}


def mechanism_summary(metadata, primary, stable):
    groups = defaultdict(list)
    for index, row in enumerate(metadata):
        if primary[index]:
            groups[row.get("event_stratum", "d3_original")].append(index)
    return {name: {"primary_misses": len(indices),
                   "sign_stable_fraction": float(np.mean(stable[indices]))}
            for name, indices in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    inputs = protocol["inputs"]
    for path_key, hash_key in (("manifest", "manifest_sha256"), ("targets", "targets_sha256"),
                               ("d15_result", "d15_result_sha256")):
        if sha256(ROOT / inputs[path_key]) != inputs[hash_key]:
            raise SystemExit(f"frozen input hash differs: {inputs[path_key]}")
    manifest = json.loads((ROOT / inputs["manifest"]).read_text())
    directories = {"d3": ROOT / inputs["d3_directory"],
                   "d12": ROOT / inputs["d12_directory"]}
    folds = {fold: load_records(manifest, directories, {fold}) for fold in
             ("fit", "episode_validation", "configuration_validation")}
    for fold, expected in inputs["groups"].items():
        if len(folds[fold][0]) != expected:
            raise SystemExit(f"{fold} group count differs")
    # Materialize fit normalizers as in D13 to preserve the frozen data path. Commitment itself
    # uses raw traces (see evidence_for_group), so these values are not applied here.
    response_normalizers(*folds["fit"][:3])
    target_file = np.load(ROOT / inputs["targets"], allow_pickle=False)
    targets = {fold: target_file[f"{fold}_raw"][:, 2] for fold in folds}; target_file.close()
    evidence = {fold: evidence_for_fold(values[1], values[2])
                for fold, values in folds.items()}
    for fold, (baseline, _, _) in evidence.items():
        if not np.allclose(baseline, targets[fold], atol=2e-4, rtol=1e-5):
            raise SystemExit(f"{fold} baseline commitment does not reproduce D13 targets")
    variant_names = evidence["fit"][2]
    if any(row[2] != variant_names for row in evidence.values()):
        raise SystemExit("variant ordering differs across folds")
    fit_margin = np.abs(evidence["fit"][0]); d15 = json.loads((ROOT / inputs["d15_result"]).read_text())
    axes = {}
    for fold in ("episode_validation", "configuration_validation"):
        baseline, variants, _ = evidence[fold]; rows = d15["axes"][fold]["rows"]
        if len(rows) != len(baseline):
            raise SystemExit(f"D15 ordering differs for {fold}")
        reference_positive = baseline > 0
        predicted_positive = np.asarray([row["predicted_positive"] for row in rows], bool)
        if any(bool(row["reference_positive"]) != bool(reference_positive[index])
               for index, row in enumerate(rows)):
            raise SystemExit(f"D15 target signs differ for {fold}")
        primary = reference_positive & ~predicted_positive
        control = reference_positive & predicted_positive
        margin_percentile = percentile_against_fit(np.abs(baseline), fit_margin)
        signs = variants > 0; stable = np.all(signs == reference_positive[:, None], axis=1)
        subsets = {"primary_misses": subset_summary(
                       primary, baseline, variants, margin_percentile),
                   "positive_correct": subset_summary(
                       control, baseline, variants, margin_percentile),
                   "all_positive": subset_summary(
                       reference_positive, baseline, variants, margin_percentile),
                   "all_groups": subset_summary(
                       np.ones(len(baseline), bool), baseline, variants, margin_percentile)}
        primary_unstable = subsets["primary_misses"]["sign_unstable_fraction"]
        control_unstable = subsets["positive_correct"]["sign_unstable_fraction"]
        axes[fold] = {"groups": len(baseline), "subsets": subsets,
                      "primary_instability_minus_control": primary_unstable - control_unstable,
                      "mechanisms": mechanism_summary(folds[fold][3], primary, stable),
                      "variant_positive_fraction": {
                          name: float(np.mean(variants[:, index] > 0))
                          for index, name in enumerate(variant_names)}}
    brittle = all(row["subsets"]["primary_misses"]["sign_unstable_fraction"] >= .5 and
                  row["primary_instability_minus_control"] >= .2 for row in axes.values())
    robust = all(row["subsets"]["primary_misses"]["sign_stable_fraction"] >= .8
                 for row in axes.values())
    interpretation = ("brittle_target" if brittle else
                      "robust_nonlinear_geometry" if robust else "mixed")
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path), "variant_names": variant_names,
              "baseline_reproduces_d13_targets": True, "axes": axes,
              "interpretation": interpretation,
              "model_training_or_threshold_selection": False,
              "joint_dev_or_test_read": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "variant_names", "baseline_reproduces_d13_targets", "axes", "interpretation",
        "model_training_or_threshold_selection", "joint_dev_or_test_read")}, indent=2))


if __name__ == "__main__":
    main()
