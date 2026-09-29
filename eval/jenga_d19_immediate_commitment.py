"""Frozen D19 comparison of full-trajectory and H8+5 immediate commitment."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from consequence_monitor import (commitment_delta_bic, persistence_delta_bic,
                                 whole_trajectory_separation_curves)
from jenga_d6_set_response import sha256
from jenga_d12_model_ladder import load_records
from jenga_d18_commitment_robustness import robustness_curves
from jenga_generic_regime_audit import reconstruct_endpoints


PROTOCOL = ROOT / "results/jenga/d19_immediate_commitment_protocol.json"
REPORT = ROOT / "results/jenga/d19_immediate_commitment_result.json"
HORIZON = 8
IMMEDIATE_HOLD = 5
IMMEDIATE_LENGTH = 1 + HORIZON + IMMEDIATE_HOLD


def commitment_values(actions, traces):
    curve = whole_trajectory_separation_curves(traces[-1:, :, :, :45])[0]
    difference = actions[-1, 0, :HORIZON] - actions[-1, 1, :HORIZON]
    full = commitment_delta_bic(curve, difference, HORIZON, IMMEDIATE_HOLD)[0]
    immediate = commitment_delta_bic(
        curve[:IMMEDIATE_LENGTH], difference, HORIZON, IMMEDIATE_HOLD)[0]
    persistence = persistence_delta_bic(curve[HORIZON:], 10)
    return float(full), float(immediate), float(persistence)


def late_sampling_invariant(curve):
    """Verify that every frozen D18 late-tail perturbation leaves D19's input unchanged."""
    prefix = np.asarray(curve)[:IMMEDIATE_LENGTH]
    return all(np.array_equal(np.asarray(values)[:IMMEDIATE_LENGTH], prefix)
               for values in robustness_curves(curve).values())


def phi(left, right):
    left = np.asarray(left, bool); right = np.asarray(right, bool)
    a = np.sum(left & right); b = np.sum(left & ~right)
    c = np.sum(~left & right); d = np.sum(~left & ~right)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float((a * d - b * c) / denominator) if denominator else 0.


def formulation_summary(full, immediate, persistence):
    full_positive = np.asarray(full) > 0; immediate_positive = np.asarray(immediate) > 0
    persistent = np.asarray(persistence) > 0
    positives = int(full_positive.sum()); negatives = int((~full_positive).sum())
    retained = int(np.sum(full_positive & immediate_positive))
    added = int(np.sum(~full_positive & immediate_positive))
    return {"groups": len(full_positive), "full_positive": positives,
            "immediate_positive": int(immediate_positive.sum()),
            "retained_full_positive": retained,
            "full_positive_retention": retained / positives if positives else 1.,
            "added_positive": added,
            "added_positive_rate_among_full_negatives": added / negatives if negatives else 0.,
            "full_persistence_phi": phi(full_positive, persistent),
            "immediate_persistence_phi": phi(immediate_positive, persistent),
            "full_and_persistent": int(np.sum(full_positive & persistent)),
            "immediate_and_persistent": int(np.sum(immediate_positive & persistent))}


def mechanism_summary(metadata, full, immediate):
    grouped = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped[row.get("event_stratum", "d3_original")].append(index)
    output = {}
    for name, indices in sorted(grouped.items()):
        indices = np.asarray(indices); old = np.asarray(full)[indices] > 0
        new = np.asarray(immediate)[indices] > 0
        output[name] = {"groups": len(indices), "full_positive": int(old.sum()),
                        "immediate_positive": int(new.sum()),
                        "retained": int(np.sum(old & new))}
    return output


def stage1(protocol, folds, frozen_targets):
    axes = {}
    invariance = {}
    for fold, values in folds.items():
        curves = [whole_trajectory_separation_curves(trace[-1:, :, :, :45])[0]
                  for trace in values[2]]
        invariance[fold] = all(late_sampling_invariant(curve) for curve in curves)
        evidence = np.asarray([commitment_values(action, trace)
                               for action, trace in zip(values[1], values[2])])
        full, immediate, persistence = evidence.T
        if not np.allclose(full, frozen_targets[fold], atol=2e-4, rtol=1e-5):
            raise RuntimeError(f"{fold} full commitment does not reproduce D13")
        summary = formulation_summary(full, immediate, persistence)
        row = {"summary": summary, "mechanisms": mechanism_summary(values[3], full, immediate),
               "full_delta_bic": full.tolist(), "immediate_delta_bic": immediate.tolist(),
               "persistence_delta_bic": persistence.tolist()}
        axes[fold] = row
    gate_cfg = protocol["stage1_gate_on_each_primary_axis"]
    gates = {}
    for fold in ("episode_validation", "configuration_validation"):
        summary = axes[fold]["summary"]
        checks = {"positive_retention": summary["full_positive_retention"] >=
                                        gate_cfg["minimum_full_positive_retention"],
                  "added_positive_rate": summary["added_positive_rate_among_full_negatives"] <=
                                         gate_cfg["maximum_added_positive_rate_among_full_negatives"],
                  "late_sampling_invariance": invariance[fold]}
        gates[fold] = {"checks": checks, "passes": all(checks.values())}
    return axes, gates, all(row["passes"] for row in gates.values())


def pair_actions(snippets, details):
    return reconstruct_endpoints(snippets, details)


def evaluate_panel(rows, snippets, cohort_key):
    evaluated = []
    for row in rows:
        old_alarm = bool(row["alarm"]); new_alarm = False
        if "curves" in row:
            endpoints = pair_actions(snippets, row["pair_details"])
            differences = endpoints[:, 0] - endpoints[:, 1]
            curves = np.asarray(row["curves"], np.float64)
            full = np.asarray([commitment_delta_bic(curve, action, HORIZON, IMMEDIATE_HOLD)[0]
                               for curve, action in zip(curves, differences)])
            if not np.allclose(full, row["commitment_delta_bic"], atol=2e-6, rtol=1e-6):
                raise RuntimeError("stored panel commitment cannot be reproduced")
            immediate = np.asarray([commitment_delta_bic(
                curve[:IMMEDIATE_LENGTH], action, HORIZON, IMMEDIATE_HOLD)[0]
                for curve, action in zip(curves, differences)])
            boundary = ((np.asarray(row["boundary_evidence"]["early_delta_bic"]) > 0) &
                        (np.asarray(row["boundary_evidence"]["full_delta_bic"]) > 0))
            persistence = np.asarray(row["persistence_delta_bic"]) > 0
            old_rebuilt = int(np.sum(boundary & (full > 0) & persistence)) >= 2
            if old_rebuilt != old_alarm:
                raise RuntimeError("stored panel alarm cannot be reproduced")
            new_alarm = bool(np.sum(boundary & (immediate > 0) & persistence) >= 2)
            commitment = immediate.tolist()
        else:
            commitment = []
        evaluated.append({"cohort": row[cohort_key], "old_alarm": old_alarm,
                          "immediate_alarm": new_alarm,
                          "immediate_commitment_delta_bic": commitment})
    cohorts = sorted({row["cohort"] for row in evaluated})
    summary = {cohort: {"states": sum(row["cohort"] == cohort for row in evaluated),
                        "full_alarms": sum(row["cohort"] == cohort and row["old_alarm"]
                                           for row in evaluated),
                        "immediate_alarms": sum(row["cohort"] == cohort and row["immediate_alarm"]
                                                for row in evaluated)}
               for cohort in cohorts}
    return summary, evaluated


def stage2(protocol):
    sources = protocol["stage2_inputs_if_stage1_passes"]
    for path_key, hash_key in (("jenga_result", "jenga_result_sha256"),
                               ("jenga_snippets", "jenga_snippets_sha256"),
                               ("push_result", "push_result_sha256"),
                               ("push_cache", "push_cache_sha256")):
        if sha256(ROOT / sources[path_key]) != sources[hash_key]:
            raise RuntimeError(f"stage-two frozen input differs: {sources[path_key]}")
    jenga = json.loads((ROOT / sources["jenga_result"]).read_text())
    with np.load(ROOT / sources["jenga_snippets"], allow_pickle=False) as cache:
        jenga_snippets = cache["snippets"]
    jenga_rows = [{**row, "cohort": row["class"]} for row in jenga["rows"]]
    jenga_summary, jenga_evidence = evaluate_panel(jenga_rows, jenga_snippets, "cohort")
    push = json.loads((ROOT / sources["push_result"]).read_text())
    with np.load(ROOT / sources["push_cache"], allow_pickle=False) as cache:
        push_snippets = cache["snippets"]
    push_summary, push_evidence = evaluate_panel(push["rows"], push_snippets, "cohort")
    gate_cfg = protocol["stage2_gate"]
    checks = {
        "jenga_topple": jenga_summary["topple_fork"]["immediate_alarms"] >=
                         gate_cfg["jenga_topple_fork_minimum_alarms"],
        "jenga_quiet": jenga_summary["quiet"]["immediate_alarms"] <=
                        gate_cfg["jenga_quiet_maximum_alarms"],
        "push_mixed_topple": push_summary["topple_fork"]["immediate_alarms"] >=
                             gate_cfg["push_mixed_topple_minimum_alarms"],
        "push_stable": push_summary["safe_centered"]["immediate_alarms"] <=
                       gate_cfg["push_stable_centered_maximum_alarms"],
        "push_overshoot": push_summary["overshoot_control"]["immediate_alarms"] <=
                          gate_cfg["push_overshoot_maximum_alarms"]}
    return {"jenga": {"summary": jenga_summary, "rows": jenga_evidence},
            "push": {"summary": push_summary, "rows": push_evidence},
            "gate": {"checks": checks, "passes": all(checks.values())}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    inputs = protocol["train_inputs"]
    for path_key, hash_key in (("manifest", "manifest_sha256"), ("targets", "targets_sha256"),
                               ("d18_result", "d18_result_sha256")):
        if sha256(ROOT / inputs[path_key]) != inputs[hash_key]:
            raise SystemExit(f"frozen input hash differs: {inputs[path_key]}")
    manifest = json.loads((ROOT / inputs["manifest"]).read_text())
    directories = {"d3": ROOT / inputs["d3_directory"], "d12": ROOT / inputs["d12_directory"]}
    folds = {fold: load_records(manifest, directories, {fold}) for fold in
             ("fit", "episode_validation", "configuration_validation")}
    target_file = np.load(ROOT / inputs["targets"], allow_pickle=False)
    targets = {fold: target_file[f"{fold}_raw"][:, 2] for fold in folds}; target_file.close()
    axes, gates, stage1_passes = stage1(protocol, folds, targets)
    panels = stage2(protocol) if stage1_passes else None
    stage2_passes = bool(panels and panels["gate"]["passes"])
    decision = ("stage1_fail_retain_v0" if not stage1_passes else
                "stage2_fail_retain_v0" if not stage2_passes else
                "both_pass_promote_immediate_commitment_reference")
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path), "stage1": {"axes": axes, "gates": gates,
              "passes": stage1_passes}, "stage2": panels, "stage2_executed": panels is not None,
              "stage2_passes": stage2_passes, "decision": decision,
              "model_training_or_threshold_selection": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage1": {"axes": {fold: row["summary"] for fold, row in axes.items()},
                                          "gates": gates, "passes": stage1_passes},
                      "stage2": None if panels is None else {
                          "jenga": panels["jenga"]["summary"],
                          "push": panels["push"]["summary"], "gate": panels["gate"]},
                      "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
