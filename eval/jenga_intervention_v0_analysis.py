"""Descriptive paired analysis of the immutable intervention-v0 experiment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_intervention_v0 import OUTPUT as RESULT, sha256_file, verify  # noqa: E402

ANALYSIS = ROOT / "results/jenga/intervention_v0/analysis.json"
ARMS = ("reobserve", "wrapper", "oracle")
METRICS = ("neighbor_failure", "pick_success", "safe_completion")
RESAMPLES = 10000
SEED = 0


def paired_comparison(rows, reference, arm, metric, indices):
    baseline = np.asarray([row["arms"][reference][metric] for row in rows], int)
    treatment = np.asarray([row["arms"][arm][metric] for row in rows], int)
    difference = treatment - baseline
    bootstrap = difference[indices].mean(1)
    reference_only = int(np.sum((baseline == 1) & (treatment == 0)))
    treatment_only = int(np.sum((baseline == 0) & (treatment == 1)))
    discordant = reference_only + treatment_only
    p = (float(binomtest(min(reference_only, treatment_only), discordant, .5).pvalue)
         if discordant else 1.0)
    return {
        "reference": reference, "arm": arm, "metric": metric,
        "reference_rate": float(baseline.mean()), "arm_rate": float(treatment.mean()),
        "arm_minus_reference": float(difference.mean()),
        "paired_bootstrap_95": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "reference_only": reference_only, "arm_only": treatment_only,
        "exact_mcnemar_p_two_sided": p,
    }


def intervention_diagnostics(rows):
    out = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        decisions = [decision for value in values for decision in value["decisions"]]
        persistent = [decision for decision in decisions if decision["persistent_alarm"]]
        entry = {
            "decision_points": len(decisions),
            "episodes_intervened": int(sum(value["reobservations"] > 0 for value in values)),
            "initial_alarms_reobservations": int(sum(value["reobservations"] for value in values)),
            "persistent_alarms": len(persistent),
            "cleared_by_one_reobservation": int(sum(
                decision["initial_alarm"] and not decision["persistent_alarm"]
                for decision in decisions)),
            "modified_chunks": int(sum(value["modified_chunks"] for value in values)),
        }
        if arm in ("wrapper", "oracle"):
            scales = [float(decision["chosen_scale"]) for decision in persistent]
            entry["persistent_choice_counts"] = {
                str(scale): int(sum(value == scale for value in scales))
                for scale in (1.0, .75, .5, .25, 0.0)}
        if arm == "wrapper":
            original = np.asarray([decision["candidate_consequential_pairs"][0]
                                   for decision in persistent], int)
            chosen = np.asarray([min(decision["candidate_consequential_pairs"])
                                 for decision in persistent], int)
            entry["candidate_reduced_pair_count"] = int(np.sum(chosen < original))
            entry["candidate_reached_zero_pairs"] = int(np.sum(chosen == 0))
        out[arm] = entry
    return out


def analyse(result):
    rows = result["rows"]
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(rows), size=(RESAMPLES, len(rows)))
    versus_baseline = {
        arm: {metric: paired_comparison(rows, "no_monitor", arm, metric, indices)
              for metric in METRICS} for arm in ARMS}
    wrapper_vs_reobserve = {
        metric: paired_comparison(rows, "reobserve", "wrapper", metric, indices)
        for metric in METRICS}
    baseline_failure_gain = -versus_baseline["wrapper"]["neighbor_failure"][
        "arm_minus_reference"]
    oracle_failure_gain = -versus_baseline["oracle"]["neighbor_failure"][
        "arm_minus_reference"]
    baseline_completion_gain = versus_baseline["wrapper"]["safe_completion"][
        "arm_minus_reference"]
    oracle_completion_gain = versus_baseline["oracle"]["safe_completion"][
        "arm_minus_reference"]
    return {
        "selection_policy": "descriptive only; frozen detector and intervention policy unchanged",
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "episode"},
        "versus_no_monitor": versus_baseline,
        "wrapper_versus_reobserve": wrapper_vs_reobserve,
        "intervention_diagnostics": intervention_diagnostics(rows),
        "fraction_of_oracle_net_gain": {
            "neighbor_failure_reduction": float(baseline_failure_gain / oracle_failure_gain),
            "safe_completion_increase": float(baseline_completion_gain / oracle_completion_gain),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ANALYSIS))
    args = parser.parse_args()
    verify()
    output = Path(args.output)
    if output.resolve() != ANALYSIS.resolve():
        raise SystemExit(f"analysis must be written to {ANALYSIS}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    result = json.loads(RESULT.read_text())
    analysis = analyse(result)
    analysis.update({"source": str(RESULT.relative_to(ROOT)),
                     "source_sha256": sha256_file(RESULT)})
    output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
