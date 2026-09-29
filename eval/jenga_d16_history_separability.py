"""Read-only D16 audit: does structured causal history resolve D15 aliases?"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256
from jenga_d12_model_ladder import load_records, response_normalizers
from jenga_d14_object_graph import permute_objects
from jenga_d15_alias_audit import (action_vectors, graph_vectors, neighbor_metrics,
                                   pairwise_rms)
from set_response_model import ObjectGraphJointMonitorEvidenceModel


PROTOCOL = ROOT / "results/jenga/d16_history_separability_protocol.json"
REPORT = ROOT / "results/jenga/d16_history_separability_result.json"


def history_lookup(path):
    with np.load(path, allow_pickle=False) as data:
        return {(str(source), int(row)): (data["state_history"][index],
                                          data["action_history"][index])
                for index, (source, row) in enumerate(zip(data["source"], data["source_row"]))}


def histories_for(metadata, lookups):
    values = []
    for row in metadata:
        key = (Path(row["file"]).stem, int(row["index"]))
        values.append(lookups[row["dataset"]][key])
    return (np.asarray([value[0] for value in values], np.float32),
            np.asarray([value[1] for value in values], np.float32))


def history_control_vectors(states, actions, action_scale):
    controls = np.asarray(actions, np.float64).copy()
    controls[..., :3] -= np.asarray(states, np.float64)[:, :-1, 57:60]
    pooled_scale = np.sqrt(np.mean(np.asarray(action_scale[:8], np.float64) ** 2, axis=0))
    controls /= np.maximum(pooled_scale, 1e-8)
    return controls.reshape(len(controls), -1)


def transition_vectors(graph_type, histories, controls, action_scale, device, order=None):
    values = torch.as_tensor(histories, device=device)
    if order is not None:
        flat = values.reshape(-1, 61)
        values = permute_objects(flat, order).reshape_as(values)
    state_vectors = graph_vectors(graph_type, values.reshape(-1, 61), device).reshape(
        len(values), values.shape[1], -1)
    transition = np.diff(state_vectors, axis=1).reshape(len(values), -1)
    control = history_control_vectors(histories, controls, action_scale)
    return np.concatenate([transition, control], axis=1)


def history_context_distance(fit_history, fit_controls, held_history, held_controls,
                             action_scale, device):
    graph_type = ObjectGraphJointMonitorEvidenceModel
    fit = transition_vectors(graph_type, fit_history, fit_controls, action_scale, device)
    candidates = []
    for order in itertools.permutations(range(3)):
        held = transition_vectors(graph_type, held_history, held_controls,
                                  action_scale, device, order)
        candidates.append(pairwise_rms(held, fit))
    return np.min(np.stack(candidates), axis=0)


def snapshot_distance(fit, held, action_scale, device):
    graph_type = ObjectGraphJointMonitorEvidenceModel
    fit_scene = graph_vectors(graph_type, fit[0], device)
    scene_candidates = []
    held_tensor = torch.as_tensor(held[0], device=device)
    for order in itertools.permutations(range(3)):
        held_scene = graph_vectors(graph_type, permute_objects(held_tensor, order), device)
        scene_candidates.append(pairwise_rms(held_scene, fit_scene))
    scene = np.min(np.stack(scene_candidates), axis=0)
    action = pairwise_rms(action_vectors(held[1], held[0], action_scale),
                          action_vectors(fit[1], fit[0], action_scale))
    return np.sqrt((scene ** 2 + action ** 2) * .5)


def summary(metrics, mask):
    mask = np.asarray(mask, bool)
    return {"count": int(mask.sum()),
            "nearest_distance_median": float(np.median(metrics["nearest_distance"][mask])),
            "same_sign_purity_median": float(np.median(metrics["same_sign_purity"][mask])),
            "opposite_to_same_ratio_median": float(np.median(
                metrics["opposite_to_same_ratio"][mask])),
            "five_neighbor_accuracy": float(np.mean(metrics["five_neighbor_correct"][mask]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    inputs = protocol["inputs"]
    hash_inputs = (("manifest", "manifest_sha256"), ("targets", "targets_sha256"),
                   ("d15_result", "d15_result_sha256"),
                   ("d3_history", "d3_history_sha256"),
                   ("d12_history", "d12_history_sha256"),
                   ("d3_alignment_result", "d3_alignment_result_sha256"),
                   ("d12_alignment_result", "d12_alignment_result_sha256"))
    for path_key, hash_key in hash_inputs:
        if sha256(ROOT / inputs[path_key]) != inputs[hash_key]:
            raise SystemExit(f"frozen input hash differs: {inputs[path_key]}")
    for key in ("d3_alignment_result", "d12_alignment_result"):
        if not json.loads((ROOT / inputs[key]).read_text())["gate"]["passes"]:
            raise SystemExit(f"history alignment does not pass: {inputs[key]}")

    manifest = json.loads((ROOT / inputs["manifest"]).read_text())
    directories = {"d3": ROOT / "results/jenga/d3_data",
                   "d12": ROOT / "results/jenga/d12_coverage_data"}
    folds = {fold: load_records(manifest, directories, {fold}) for fold in
             ("fit", "episode_validation", "configuration_validation")}
    for fold, expected in (("fit", inputs["fit_groups"]),
                           ("episode_validation", inputs["episode_validation_groups"]),
                           ("configuration_validation", inputs["configuration_validation_groups"])):
        if len(folds[fold][0]) != expected:
            raise SystemExit(f"{fold} group count differs")
    lookups = {"d3": history_lookup(ROOT / inputs["d3_history"]),
               "d12": history_lookup(ROOT / inputs["d12_history"])}
    histories = {fold: histories_for(values[3], lookups) for fold, values in folds.items()}
    for fold, values in folds.items():
        if not np.array_equal(histories[fold][0][:, -1], values[0]):
            raise SystemExit(f"{fold} history current state differs from response start")
    normalizer_np = response_normalizers(*folds["fit"][:3]); action_scale = normalizer_np[3]
    target_file = np.load(ROOT / inputs["targets"], allow_pickle=False)
    targets = {fold: target_file[f"{fold}_raw"][:, 2] for fold in folds}; target_file.close()
    fit_positive = targets["fit"] > 0
    d15 = json.loads((ROOT / inputs["d15_result"]).read_text())
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    k = protocol["history_representation"]["neighbors"]; axes = {}
    for fold in ("episode_validation", "configuration_validation"):
        held_positive = targets[fold] > 0
        rows = d15["axes"][fold]["rows"]
        if len(rows) != len(held_positive) or any(
                bool(row["reference_positive"]) != bool(held_positive[index])
                for index, row in enumerate(rows)):
            raise SystemExit(f"D15 row ordering differs for {fold}")
        primary = np.asarray([row["reference_positive"] and not row["predicted_positive"]
                              for row in rows], bool)
        snapshot = snapshot_distance(folds["fit"], folds[fold], action_scale, device)
        context = history_context_distance(
            histories["fit"][0], histories["fit"][1], histories[fold][0], histories[fold][1],
            action_scale, device)
        augmented = np.sqrt((snapshot ** 2 + context ** 2) * .5)
        snapshot_metrics = neighbor_metrics(snapshot, fit_positive, held_positive, k)
        context_metrics = neighbor_metrics(context, fit_positive, held_positive, k)
        augmented_metrics = neighbor_metrics(augmented, fit_positive, held_positive, k)
        snapshot_primary = summary(snapshot_metrics, primary)
        context_primary = summary(context_metrics, primary)
        augmented_primary = summary(augmented_metrics, primary)
        snapshot_all = summary(snapshot_metrics, np.ones(len(primary), bool))
        augmented_all = summary(augmented_metrics, np.ones(len(primary), bool))
        gate_cfg = protocol["gate_on_each_primary_axis"]
        comparisons = {
            "primary_error_same_sign_purity_gain": (
                augmented_primary["same_sign_purity_median"]
                - snapshot_primary["same_sign_purity_median"]),
            "primary_error_opposite_to_same_ratio":
                augmented_primary["opposite_to_same_ratio_median"],
            "overall_five_neighbor_accuracy_change": (
                augmented_all["five_neighbor_accuracy"] - snapshot_all["five_neighbor_accuracy"])}
        checks = {
            "purity_gain": comparisons["primary_error_same_sign_purity_gain"] >=
                           gate_cfg["minimum_primary_error_same_sign_purity_gain"],
            "opposite_separation": comparisons["primary_error_opposite_to_same_ratio"] >=
                                   gate_cfg["minimum_primary_error_opposite_to_same_ratio"],
            "overall_accuracy": comparisons["overall_five_neighbor_accuracy_change"] >=
                                -gate_cfg["maximum_overall_five_neighbor_accuracy_loss"]}
        axes[fold] = {"primary_errors": int(primary.sum()),
                      "snapshot_primary": snapshot_primary,
                      "history_context_primary": context_primary,
                      "augmented_primary": augmented_primary,
                      "snapshot_all": snapshot_all, "augmented_all": augmented_all,
                      "comparisons": comparisons, "checks": checks,
                      "passes": all(checks.values())}
    passes = all(row["passes"] for row in axes.values())
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path), "device": device,
              "history_alignment_exact": True, "axes": axes,
              "temporal_graph_training_authorized": passes,
              "decision": ("history_resolves_aliases_authorize_temporal_graph" if passes else
                           "history_does_not_resolve_aliases_collect_richer_state"),
              "model_training_or_threshold_selection": False,
              "joint_dev_or_test_read": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "history_alignment_exact", "axes", "temporal_graph_training_authorized", "decision",
        "model_training_or_threshold_selection", "joint_dev_or_test_read")}, indent=2))


if __name__ == "__main__":
    main()
