"""Read-only D15 audit of D14 held-out commitment errors and representation aliases."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256, tensors
from jenga_d12_model_ladder import load_records, response_normalizers
from jenga_d13_monitor_evidence import decode_evidence
from jenga_d14_object_graph import permute_objects
from set_response_model import ObjectGraphJointMonitorEvidenceModel


PROTOCOL = ROOT / "results/jenga/d15_alias_audit_protocol.json"
REPORT = ROOT / "results/jenga/d15_alias_audit_result.json"


def pairwise_rms(left, right):
    left = np.asarray(left, np.float64); right = np.asarray(right, np.float64)
    return np.sqrt(np.mean((left[:, None] - right[None]) ** 2, axis=-1))


def graph_vectors(model, states, device):
    with torch.no_grad():
        objects, edges, gripper, _, _ = model.graph_inputs(
            torch.as_tensor(states, device=device))
    return np.concatenate([objects.cpu().numpy().reshape(len(states), -1),
                           edges.cpu().numpy().reshape(len(states), -1),
                           gripper.cpu().numpy()], axis=1)


def permutation_invariant_scene_distance(model, held_states, fit_vectors, device):
    distances = []
    tensor = torch.as_tensor(held_states, device=device)
    for order in itertools.permutations(range(3)):
        vectors = graph_vectors(model, permute_objects(tensor, order), device)
        distances.append(pairwise_rms(vectors, fit_vectors))
    return np.min(np.stack(distances), axis=0)


def action_vectors(actions, states, action_scale):
    control = np.asarray(actions, np.float64)[..., :8, :].copy()
    control[..., :3] -= np.asarray(states, np.float64)[:, None, None, None, 57:60]
    control[..., :3] /= np.maximum(np.asarray(action_scale[:8, :3]), 1e-8)
    control[..., 3:] /= np.maximum(np.asarray(action_scale[:8, 3:]), 1e-8)
    left, right = control[:, :, 0], control[:, :, 1]
    return np.concatenate([(left + right) * .5, np.abs(left - right)], axis=-1).reshape(
        len(actions), -1)


def model_values(model, data, norms, target_mean, target_scale, device, batch_size=128):
    embeddings, evidence = [], []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(data[0]), batch_size):
            state = torch.as_tensor(data[0][first:first + batch_size], device=device)
            action = torch.as_tensor(data[1][first:first + batch_size], device=device)
            pair = model.encode_pairs(state, action, *norms[:4])
            embeddings.append(torch.cat([pair.mean(1), pair.max(1).values], -1).cpu().numpy())
            output = model(state, action, *norms[:4])[-1]
            evidence.append(output.cpu().numpy())
    prediction = decode_evidence(np.concatenate(evidence), target_mean, target_scale)
    return np.concatenate(embeddings), prediction


def neighbor_metrics(distance, fit_positive, held_positive, neighbors):
    order = np.argsort(distance, axis=1, kind="mergesort")
    nearest = order[:, 0]; local = order[:, :neighbors]
    purity = np.mean(fit_positive[local] == held_positive[:, None], axis=1)
    same = np.where(fit_positive[None] == held_positive[:, None], distance, np.inf)
    opposite = np.where(fit_positive[None] != held_positive[:, None], distance, np.inf)
    same_distance = np.min(same, axis=1); opposite_distance = np.min(opposite, axis=1)
    ratio = opposite_distance / np.maximum(same_distance, 1e-12)
    vote = np.mean(fit_positive[local], axis=1) > .5
    return {"nearest_index": nearest, "nearest_distance": distance[np.arange(len(distance)), nearest],
            "same_sign_purity": purity, "nearest_same_distance": same_distance,
            "nearest_opposite_distance": opposite_distance,
            "opposite_to_same_ratio": ratio, "five_neighbor_vote": vote,
            "five_neighbor_correct": vote == held_positive}


def median(values, mask):
    selected = np.asarray(values)[mask]
    return float(np.median(selected)) if len(selected) else None


def subset_summary(mask, raw, learned, target, predicted):
    mask = np.asarray(mask, bool)
    return {"count": int(mask.sum()),
            "raw_nearest_distance_median": median(raw["nearest_distance"], mask),
            "raw_same_sign_purity_median": median(raw["same_sign_purity"], mask),
            "raw_opposite_to_same_ratio_median": median(raw["opposite_to_same_ratio"], mask),
            "learned_nearest_distance_median": median(learned["nearest_distance"], mask),
            "learned_same_sign_purity_median": median(learned["same_sign_purity"], mask),
            "learned_opposite_to_same_ratio_median": median(
                learned["opposite_to_same_ratio"], mask),
            "absolute_commitment_error_median": median(np.abs(predicted - target), mask),
            "raw_five_neighbor_accuracy": (float(np.mean(raw["five_neighbor_correct"][mask]))
                                             if mask.any() else None),
            "learned_five_neighbor_accuracy": (
                float(np.mean(learned["five_neighbor_correct"][mask])) if mask.any() else None)}


def diagnostic_marker(summary):
    raw_purity = summary["raw_same_sign_purity_median"]
    learned_purity = summary["learned_same_sign_purity_median"]
    ratio = summary["raw_opposite_to_same_ratio_median"]
    if summary["count"] == 0:
        return "no_primary_errors"
    if raw_purity < .60 and ratio <= 1.25:
        return "raw_snapshot_alias_pressure"
    if raw_purity >= .80 and learned_purity < .60:
        return "learned_representation_collapse"
    if raw_purity >= .80 and learned_purity >= .60:
        return "mapping_or_coverage_failure"
    return "mixed"


def stratum_summaries(metadata, target_positive, predicted_positive):
    grouped = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped[row.get("event_stratum", "d3_original")].append(index)
    output = {}
    for name, indices in sorted(grouped.items()):
        indices = np.asarray(indices, int); truth = target_positive[indices]
        prediction = predicted_positive[indices]
        positive = int(truth.sum()); negative = int((~truth).sum())
        output[name] = {"groups": len(indices), "reference_positive": positive,
                        "commitment_recall": (float(np.mean(prediction[truth]))
                                              if positive else None),
                        "added_positive_rate": (float(np.mean(prediction[~truth]))
                                                if negative else None)}
    return output


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
    for relative, expected in (
            (inputs["manifest"], inputs["manifest_sha256"]),
            (inputs["targets"], inputs["targets_sha256"]),
            (inputs["d14_result"], inputs["d14_result_sha256"]),
            (inputs["d14_checkpoint"], inputs["d14_checkpoint_sha256"])):
        if sha256(ROOT / relative) != expected:
            raise SystemExit(f"frozen input hash differs: {relative}")
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
    normalizer_np = response_normalizers(*folds["fit"][:3])
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    norms = tensors(normalizer_np, device)
    target_file = np.load(ROOT / inputs["targets"], allow_pickle=False)
    target_mean = target_file["target_mean"]; target_scale = target_file["target_scale"]
    targets = {fold: target_file[f"{fold}_raw"] for fold in folds}; target_file.close()
    d14 = json.loads((ROOT / inputs["d14_result"]).read_text())["protocol"]["model"]
    model = ObjectGraphJointMonitorEvidenceModel(
        hidden=d14["hidden"], heads=d14["heads"], layers=d14["layers"],
        graph_rounds=d14["graph_rounds"], action_horizon=d14["action_horizon"],
        curve_steps=d14["curve_steps"]).to(device)
    checkpoint = torch.load(ROOT / inputs["d14_checkpoint"], map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()

    fit = folds["fit"]; fit_vectors = graph_vectors(model, fit[0], device)
    fit_actions = action_vectors(fit[1], fit[0], normalizer_np[3])
    fit_embedding, fit_prediction = model_values(
        model, fit, norms, target_mean, target_scale, device)
    embedding_mean = fit_embedding.mean(0); embedding_scale = fit_embedding.std(0)
    embedding_scale = np.maximum(embedding_scale, max(float(embedding_scale.mean()) * 1e-3, 1e-6))
    fit_embedding_z = (fit_embedding - embedding_mean) / embedding_scale
    fit_positive = targets["fit"][:, 2] > 0
    neighbors = protocol["distances"]["neighbors"]
    axes = {}; all_markers = []
    for fold in ("episode_validation", "configuration_validation"):
        values = folds[fold]
        scene_distance = permutation_invariant_scene_distance(
            model, values[0], fit_vectors, device)
        held_actions = action_vectors(values[1], values[0], normalizer_np[3])
        action_distance = pairwise_rms(held_actions, fit_actions)
        raw_distance = np.sqrt((scene_distance ** 2 + action_distance ** 2) * .5)
        held_embedding, prediction = model_values(
            model, values, norms, target_mean, target_scale, device)
        held_embedding_z = (held_embedding - embedding_mean) / embedding_scale
        learned_distance = pairwise_rms(held_embedding_z, fit_embedding_z)
        target = targets[fold][:, 2]; predicted = prediction[:, 2]
        positive = target > 0; predicted_positive = predicted > 0
        raw = neighbor_metrics(raw_distance, fit_positive, positive, neighbors)
        learned = neighbor_metrics(learned_distance, fit_positive, positive, neighbors)
        masks = {"primary_errors": positive & ~predicted_positive,
                 "positive_correct": positive & predicted_positive,
                 "negative_added": ~positive & predicted_positive,
                 "negative_correct": ~positive & ~predicted_positive,
                 "all": np.ones(len(positive), bool)}
        summaries = {name: subset_summary(mask, raw, learned, target, predicted)
                     for name, mask in masks.items()}
        marker = diagnostic_marker(summaries["primary_errors"]); all_markers.append(marker)
        rows = []
        for index, metadata in enumerate(values[3]):
            rows.append({"index": index, "dataset": metadata["dataset"],
                         "file": metadata["file"], "source_index": metadata["index"],
                         "stratum": metadata.get("event_stratum", "d3_original"),
                         "reference_commitment": float(target[index]),
                         "predicted_commitment": float(predicted[index]),
                         "reference_positive": bool(positive[index]),
                         "predicted_positive": bool(predicted_positive[index]),
                         "raw_nearest_fit_index": int(raw["nearest_index"][index]),
                         "raw_nearest_fit_source": fit[3][int(raw["nearest_index"][index])],
                         "raw_nearest_distance": float(raw["nearest_distance"][index]),
                         "raw_same_sign_purity": float(raw["same_sign_purity"][index]),
                         "raw_opposite_to_same_ratio": float(
                             raw["opposite_to_same_ratio"][index]),
                         "learned_nearest_fit_index": int(learned["nearest_index"][index]),
                         "learned_nearest_fit_source": fit[3][int(learned["nearest_index"][index])],
                         "learned_nearest_distance": float(learned["nearest_distance"][index]),
                         "learned_same_sign_purity": float(learned["same_sign_purity"][index]),
                         "learned_opposite_to_same_ratio": float(
                             learned["opposite_to_same_ratio"][index])})
        axes[fold] = {"groups": len(positive), "commitment_confusion": {
            "tp": int(np.sum(positive & predicted_positive)),
            "fn": int(np.sum(positive & ~predicted_positive)),
            "fp": int(np.sum(~positive & predicted_positive)),
            "tn": int(np.sum(~positive & ~predicted_positive))},
            "raw_five_neighbor_accuracy": float(np.mean(raw["five_neighbor_correct"])),
            "learned_five_neighbor_accuracy": float(np.mean(learned["five_neighbor_correct"])),
            "subsets": summaries, "generic_strata": stratum_summaries(
                values[3], positive, predicted_positive),
            "diagnostic_marker": marker, "rows": rows}
    conclusion = (all_markers[0] if len(set(all_markers)) == 1 else "mixed_across_axes")
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(protocol_path),
              "fit_commitment_positive": int(fit_positive.sum()),
              "fit_commitment_nonpositive": int((~fit_positive).sum()),
              "fit_prediction_sign_accuracy": float(np.mean((fit_prediction[:, 2] > 0) == fit_positive)),
              "axes": axes, "conclusion": conclusion,
              "model_training_or_threshold_selection": False,
              "joint_dev_or_test_read": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {fold: {key: value for key, value in row.items() if key != "rows"}
               for fold, row in axes.items()}
    print(json.dumps({"fit_commitment_positive": result["fit_commitment_positive"],
                      "fit_commitment_nonpositive": result["fit_commitment_nonpositive"],
                      "axes": compact, "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
