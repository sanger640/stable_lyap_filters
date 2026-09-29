"""Evaluate frozen D17 snapshot, proprioception, contact, and combined separability arms."""
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
from jenga_d15_alias_audit import action_vectors, graph_vectors, neighbor_metrics, pairwise_rms
from set_response_model import ObjectGraphJointMonitorEvidenceModel


PROTOCOL = ROOT / "results/jenga/d17_enhanced_state_protocol.json"
DATA = ROOT / "results/jenga/d17_enhanced_state_data.npz"
INTEGRITY = ROOT / "results/jenga/d17_enhanced_state_integrity.json"
D15 = ROOT / "results/jenga/d15_alias_audit_result.json"
REPORT = ROOT / "results/jenga/d17_enhanced_separability_result.json"


def feature_normalize(fit, held):
    mean = fit.mean(axis=0); scale = fit.std(axis=0)
    floor = max(float(np.sqrt(np.mean(scale ** 2))) * 1e-3, 1e-8)
    scale = np.maximum(scale, floor)
    return (fit - mean) / scale, (held - mean) / scale


def contact_normalizers(fit):
    """Shared channel normalization within pair/table/floor/robot category types."""
    fit = np.asarray(fit)
    means, scales = [], []
    for first in (0, 3, 6, 9):
        values = fit[:, first:first + 3].reshape(-1, fit.shape[-1])
        mean = values.mean(0); scale = values.std(0)
        floor = max(float(np.sqrt(np.mean(scale ** 2))) * 1e-3, 1e-8)
        means.append(mean); scales.append(np.maximum(scale, floor))
    return np.asarray(means), np.asarray(scales)


def normalize_contacts(values, means, scales):
    output = np.asarray(values, np.float64).copy()
    for type_index, first in enumerate((0, 3, 6, 9)):
        output[:, first:first + 3] = ((output[:, first:first + 3] - means[type_index]) /
                                      scales[type_index])
    return output


def permute_contacts(values, order):
    """Relabel twelve pair/support/robot categories consistently with three objects."""
    source = np.asarray(values); output = source.copy(); order = list(order)
    pairs = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    for first, second in pairs:
        output[:, pairs[(first, second)]] = source[
            :, pairs[tuple(sorted((order[first], order[second])))] ]
    for offset in (3, 6, 9):
        output[:, offset:offset + 3] = source[:, offset:offset + 3][:, order]
    return output


def fold_indices(manifest, fold):
    return np.asarray([index for index, row in enumerate(manifest["records"])
                       if row["fold"] == fold], int)


def distance_arms(fit_state, fit_action, fit_proprio, fit_contact,
                  held_state, held_action, held_proprio, held_contact,
                  action_scale, device):
    graph_type = ObjectGraphJointMonitorEvidenceModel
    fit_scene = graph_vectors(graph_type, fit_state, device)
    fit_action_vector = action_vectors(fit_action, fit_state, action_scale)
    held_action_vector = action_vectors(held_action, held_state, action_scale)
    action_distance = pairwise_rms(held_action_vector, fit_action_vector)
    fit_proprio_z, held_proprio_z = feature_normalize(fit_proprio, held_proprio)
    proprio_distance = pairwise_rms(held_proprio_z, fit_proprio_z)
    contact_mean, contact_scale = contact_normalizers(fit_contact)
    fit_contact_z = normalize_contacts(fit_contact, contact_mean, contact_scale).reshape(
        len(fit_contact), -1)
    candidates = {name: [] for name in (
        "snapshot", "snapshot_plus_proprioception", "snapshot_plus_contact", "all_enhanced")}
    held_tensor = torch.as_tensor(held_state, device=device)
    for order in itertools.permutations(range(3)):
        held_scene = graph_vectors(graph_type, permute_objects(held_tensor, order), device)
        scene_distance = pairwise_rms(held_scene, fit_scene)
        held_contact_z = normalize_contacts(
            permute_contacts(held_contact, order), contact_mean, contact_scale).reshape(
                len(held_contact), -1)
        contact_distance = pairwise_rms(held_contact_z, fit_contact_z)
        candidates["snapshot"].append(np.sqrt((scene_distance ** 2 + action_distance ** 2) / 2))
        candidates["snapshot_plus_proprioception"].append(np.sqrt(
            (scene_distance ** 2 + action_distance ** 2 + proprio_distance ** 2) / 3))
        candidates["snapshot_plus_contact"].append(np.sqrt(
            (scene_distance ** 2 + action_distance ** 2 + contact_distance ** 2) / 3))
        candidates["all_enhanced"].append(np.sqrt(
            (scene_distance ** 2 + action_distance ** 2 + proprio_distance ** 2 +
             contact_distance ** 2) / 4))
    return {name: np.min(np.stack(values), axis=0) for name, values in candidates.items()}


def metric_summary(metrics, mask):
    mask = np.asarray(mask, bool)
    return {"count": int(mask.sum()),
            "nearest_distance_median": float(np.median(metrics["nearest_distance"][mask])),
            "same_sign_purity_median": float(np.median(metrics["same_sign_purity"][mask])),
            "opposite_to_same_ratio_median": float(np.median(
                metrics["opposite_to_same_ratio"][mask])),
            "five_neighbor_accuracy": float(np.mean(metrics["five_neighbor_correct"][mask]))}


def gate(baseline_primary, baseline_all, arm_primary, arm_all, config):
    comparisons = {
        "primary_error_same_sign_purity_gain": (
            arm_primary["same_sign_purity_median"] - baseline_primary["same_sign_purity_median"]),
        "primary_error_opposite_to_same_ratio": arm_primary["opposite_to_same_ratio_median"],
        "overall_five_neighbor_accuracy_change": (
            arm_all["five_neighbor_accuracy"] - baseline_all["five_neighbor_accuracy"])}
    checks = {"purity_gain": comparisons["primary_error_same_sign_purity_gain"] >=
                             config["minimum_primary_error_same_sign_purity_gain"],
              "opposite_separation": comparisons["primary_error_opposite_to_same_ratio"] >=
                                     config["minimum_primary_error_opposite_to_same_ratio"],
              "overall_accuracy": comparisons["overall_five_neighbor_accuracy_change"] >=
                                  -config["maximum_overall_five_neighbor_accuracy_loss"]}
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--integrity", default=str(INTEGRITY)); parser.add_argument("--d15", default=str(D15))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    integrity = json.loads(Path(args.integrity).read_text())
    if not integrity["passes"] or integrity["protocol_sha256"] != sha256(protocol_path):
        raise SystemExit("D17 enhanced-state integrity does not match protocol")
    if sha256(args.data) != integrity["output_sha256"]:
        raise SystemExit("D17 enhanced-state data hash differs")
    sources = protocol["sources"]
    for key, hash_key in (("manifest", "manifest_sha256"), ("targets", "targets_sha256")):
        if sha256(ROOT / sources[key]) != sources[hash_key]:
            raise SystemExit(f"frozen input hash differs: {sources[key]}")
    manifest = json.loads((ROOT / sources["manifest"]).read_text())
    directories = {"d3": ROOT / sources["d3_directory"], "d12": ROOT / sources["d12_directory"]}
    folds = {fold: load_records(manifest, directories, {fold}) for fold in
             ("fit", "episode_validation", "configuration_validation")}
    indices = {fold: fold_indices(manifest, fold) for fold in folds}
    with np.load(args.data, allow_pickle=False) as enhanced:
        base = enhanced["base_state"]
        if not all(np.array_equal(base[indices[fold]], folds[fold][0]) for fold in folds):
            raise SystemExit("enhanced-state ordering differs from manifest folds")
        proprio = {fold: enhanced["proprioception"][indices[fold]] for fold in folds}
        contact_all = np.concatenate([enhanced["instant_contact"],
                                      enhanced["recent_contact_impulse"]], axis=-1)
        contact = {fold: contact_all[indices[fold]] for fold in folds}
    target_file = np.load(ROOT / sources["targets"], allow_pickle=False)
    targets = {fold: target_file[f"{fold}_raw"][:, 2] for fold in folds}; target_file.close()
    fit_positive = targets["fit"] > 0
    normalizer_np = response_normalizers(*folds["fit"][:3]); action_scale = normalizer_np[3]
    d15 = json.loads(Path(args.d15).read_text())
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    k = protocol["gate_on_each_primary_axis"]["neighbors"]
    axes = {}; arm_names = tuple(protocol["separability_arms"])
    for fold in ("episode_validation", "configuration_validation"):
        held_positive = targets[fold] > 0; d15_rows = d15["axes"][fold]["rows"]
        primary = np.asarray([row["reference_positive"] and not row["predicted_positive"]
                              for row in d15_rows], bool)
        distances = distance_arms(
            folds["fit"][0], folds["fit"][1], proprio["fit"], contact["fit"],
            folds[fold][0], folds[fold][1], proprio[fold], contact[fold], action_scale, device)
        metrics = {name: neighbor_metrics(values, fit_positive, held_positive, k)
                   for name, values in distances.items()}
        summaries = {name: {"primary_errors": metric_summary(values, primary),
                            "all": metric_summary(values, np.ones(len(primary), bool))}
                     for name, values in metrics.items()}
        # Internal control: the recomputed snapshot statistics must equal D15.
        expected = d15["axes"][fold]["subsets"]["primary_errors"]
        control_matches = (abs(summaries["snapshot"]["primary_errors"]["same_sign_purity_median"]
                               - expected["raw_same_sign_purity_median"]) < 1e-12 and
                           abs(summaries["snapshot"]["primary_errors"]["opposite_to_same_ratio_median"]
                               - expected["raw_opposite_to_same_ratio_median"]) < 1e-12)
        gates = {name: gate(summaries["snapshot"]["primary_errors"],
                            summaries["snapshot"]["all"], summaries[name]["primary_errors"],
                            summaries[name]["all"], protocol["gate_on_each_primary_axis"])
                 for name in arm_names if name != "snapshot"}
        axes[fold] = {"groups": len(primary), "primary_errors": int(primary.sum()),
                      "snapshot_control_matches_d15": control_matches,
                      "arms": summaries, "gates": gates}
    passing = [name for name in arm_names if name != "snapshot" and all(
        axes[fold]["gates"][name]["passes"] for fold in axes)]
    simplest = next((name for name in (
        "snapshot_plus_proprioception", "snapshot_plus_contact", "all_enhanced")
                     if name in passing), None)
    if simplest == "snapshot_plus_contact":
        interpretation = "privileged_contact_only_passes_observability_limit"
    elif simplest:
        interpretation = "enhanced_state_pass_authorize_relational_model"
    else:
        interpretation = "no_enhanced_arm_passes_reconsider_commitment_observability"
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path), "data_sha256": sha256(args.data),
              "integrity_sha256": sha256(args.integrity), "device": device,
              "axes": axes, "passing_arms": passing, "simplest_passing_arm": simplest,
              "model_training_authorized": bool(simplest), "interpretation": interpretation,
              "model_training_or_threshold_selection": False,
              "joint_dev_or_test_read": False}
    if not all(row["snapshot_control_matches_d15"] for row in axes.values()):
        raise SystemExit("D17 snapshot control does not reproduce D15")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "axes", "passing_arms", "simplest_passing_arm", "model_training_authorized",
        "interpretation", "model_training_or_threshold_selection", "joint_dev_or_test_read")},
        indent=2))


if __name__ == "__main__":
    main()
