"""Read-only support-versus-mapping audit for fixed D9b held-out predictions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import DATA, NESTED, normalizers, sha256
from jenga_d7_explicit_curve import curve_targets
from jenga_reobservation_interface import load_nested, validation_rows
from set_response_model import GroupAmplitudeShapeTemporalCurveModel


PROTOCOL = ROOT / "results/jenga/d9b_support_audit_protocol.json"
REPORT = ROOT / "results/jenga/d9b_support_audit_result.json"
HELDOUT_RESULT = ROOT / "results/jenga/d9b_heldout_train_result.json"


def action_features(actions, mean, scale):
    """Return endpoint-symmetric features from the eight control steps visible to D9b."""
    normalized = (np.asarray(actions)[..., :8, :] - mean[:8]) / scale[:8]
    left, right = normalized[:, :, 0], normalized[:, :, 1]
    return np.concatenate([(left + right) * .5, np.abs(left - right)], axis=-1).reshape(len(actions), -1)


def pairwise_rms(left, right):
    left = np.asarray(left, float); right = np.asarray(right, float)
    return np.sqrt(np.mean((left[:, None] - right[None]) ** 2, axis=-1))


def average_ranks(values):
    values = np.asarray(values, float); order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = .5 * (first + 1 + last)
        first = last
    return ranks


def percentile_ranks(values):
    values = np.asarray(values, float)
    return average_ranks(values) / max(len(values), 1)


def spearman(left, right):
    x = average_ranks(left); y = average_ranks(right)
    x -= x.mean(); y -= y.mean(); denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else 0.


def subset_summary(indices, metrics, percentiles):
    result = {"count": int(len(indices)), "indices": list(map(int, indices))}
    for name, values in metrics.items():
        selected = np.asarray(values)[indices]
        ranked = np.asarray(percentiles[name])[indices]
        result[name] = {"median": float(np.median(selected)) if len(selected) else None,
                        "median_percentile": float(np.median(ranked)) if len(ranked) else None}
    return result


def checkpoint_predictions(protocol, checkpoints, start, actions, target, device):
    predictions = []
    for relative, expected_hash in checkpoints:
        path = ROOT / relative
        if sha256(path) != expected_hash: raise SystemExit(f"checkpoint hash differs: {relative}")
        saved = torch.load(path, map_location=device, weights_only=False)
        model = GroupAmplitudeShapeTemporalCurveModel(
            hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
            layers=protocol["model"]["layers"]).to(device)
        model.load_state_dict(saved["model"]); model.eval()
        norms = tuple(value.to(device) for value in saved["normalizers"])
        with torch.no_grad():
            curve, amplitude, _ = model(start, actions, *norms[:4])
        predictions.append({"curve": curve.cpu().numpy(),
                            "amplitude": amplitude[:, 0, 0].cpu().numpy(),
                            "curve_rmse": np.sqrt(np.mean(
                                (curve.cpu().numpy() - target) ** 2, axis=(1, 2)))})
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--heldout-result", default=str(HELDOUT_RESULT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.report)
    if output.exists() and not args.force: raise SystemExit(f"refusing to overwrite {output}")
    protocol = json.loads(Path(args.protocol).read_text())
    if sha256(args.heldout_result) != protocol["inputs"]["heldout_result_sha256"]:
        raise SystemExit("held-out result hash differs from diagnostic specification")
    heldout_result = json.loads(Path(args.heldout_result).read_text())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    source = np.load(args.data, allow_pickle=False)
    val_rows, heldout_episodes = validation_rows(source["episode_id"])
    heldout_episodes = list(map(str, heldout_episodes))
    if heldout_episodes != protocol["inputs"]["heldout_episode_ids"]:
        raise SystemExit("held-out episode split differs")
    train_rows = np.setdiff1d(np.arange(len(source["start"])), val_rows)
    norm = normalizers(source["start"][train_rows], source["actions"][train_rows],
                       source["traces"][train_rows])
    all_episodes = sorted(set(source["episode_id"].astype(str)), key=int)
    fit_episodes = [episode for episode in all_episodes if episode not in set(heldout_episodes)]
    fit_start, fit_actions, fit_truth, fit_sources = load_nested(args.nested, fit_episodes)
    held_start, held_actions, held_truth, held_sources = load_nested(args.nested, heldout_episodes)
    if len(fit_start) != protocol["inputs"]["fit_groups"] or len(held_start) != protocol["inputs"]["heldout_groups"]:
        raise SystemExit("group counts differ from diagnostic specification")

    fit_state = (fit_start - norm[0]) / norm[1]; held_state = (held_start - norm[0]) / norm[1]
    fit_action = action_features(fit_actions, norm[2], norm[3])
    held_action = action_features(held_actions, norm[2], norm[3])
    state_distance = pairwise_rms(held_state, fit_state)
    action_distance = pairwise_rms(held_action, fit_action)
    joint_distance = np.sqrt((state_distance ** 2 + action_distance ** 2) * .5)
    nearest_input = np.argmin(joint_distance, axis=1)

    fit_response = curve_targets(fit_truth, norm[4]).reshape(len(fit_truth), -1)
    held_response = curve_targets(held_truth, norm[4]).reshape(len(held_truth), -1)
    response_distance = pairwise_rms(held_response, fit_response)
    nearest_response = np.min(response_distance, axis=1)
    nearest_input_response_mismatch = response_distance[np.arange(len(held_start)), nearest_input]

    predictions = checkpoint_predictions(
        heldout_result["protocol"], protocol["inputs"]["checkpoints"],
        torch.as_tensor(held_start, device=device), torch.as_tensor(held_actions, device=device),
        held_response.reshape(len(held_truth), 6, 41), device)
    prediction_errors = np.stack([row["curve_rmse"] for row in predictions])
    median_prediction_error = np.median(prediction_errors, axis=0)

    evidence = [seed["evidence"] for seed in heldout_result["seeds"]]
    if any([row["source"] for row in seed] != held_sources for seed in evidence):
        raise SystemExit("held-out evidence ordering differs")
    unanimous_false_boundary = np.asarray([
        not evidence[0][i]["reference"]["boundary"]
        and all(seed[i]["predicted"]["boundary"] for seed in evidence)
        for i in range(len(held_sources))])
    repeated_false_zero = np.asarray([
        evidence[0][i]["target_amplitude"] > 0
        and sum(seed[i]["predicted_amplitude"] == 0 for seed in evidence) >= 2
        for i in range(len(held_sources))])
    error_union = unanimous_false_boundary | repeated_false_zero

    metrics = {"state_distance": np.min(state_distance, axis=1),
               "action_distance": np.min(action_distance, axis=1),
               "joint_input_distance": np.min(joint_distance, axis=1),
               "nearest_response_distance": nearest_response,
               "nearest_input_response_mismatch": nearest_input_response_mismatch,
               "prediction_error": median_prediction_error}
    percentiles = {name: percentile_ranks(values) for name, values in metrics.items()}
    subsets = {
        "unanimous_false_boundary": subset_summary(
            np.flatnonzero(unanimous_false_boundary), metrics, percentiles),
        "repeated_false_zero": subset_summary(
            np.flatnonzero(repeated_false_zero), metrics, percentiles),
        "union": subset_summary(np.flatnonzero(error_union), metrics, percentiles),
    }
    correlations = {
        name + "_vs_prediction_error": spearman(values, median_prediction_error)
        for name, values in metrics.items() if name != "prediction_error"}
    union = subsets["union"]
    joint_percentile = union["joint_input_distance"]["median_percentile"] or 0.
    mismatch_percentile = union["nearest_input_response_mismatch"]["median_percentile"] or 0.
    error_percentile = union["prediction_error"]["median_percentile"] or 0.
    markers = {
        "coverage_pressure": bool(joint_percentile >= .75 or
                                  correlations["joint_input_distance_vs_prediction_error"] >= .4),
        "representation_ambiguity": bool(joint_percentile < .75 and mismatch_percentile >= .75),
        "mapping_error": bool(joint_percentile < .75 and mismatch_percentile < .75
                              and error_percentile >= .75),
    }
    if markers["representation_ambiguity"]:
        conclusion = "in_support_inputs_map_to_different_responses_representation_ambiguity"
    elif markers["coverage_pressure"]:
        conclusion = "input_support_coverage_pressure"
    elif markers["mapping_error"]:
        conclusion = "locally_supported_mapping_error"
    else:
        conclusion = "mixed_or_inconclusive"

    groups = []
    for index, source_name in enumerate(held_sources):
        groups.append({"index": index, "source": source_name,
                       "episode": source_name.split("_seed")[0][2:],
                       "nearest_fit_source": fit_sources[int(nearest_input[index])],
                       **{name: float(values[index]) for name, values in metrics.items()},
                       "percentiles": {name: float(values[index])
                                       for name, values in percentiles.items()},
                       "seed_prediction_errors": prediction_errors[:, index].tolist(),
                       "unanimous_false_boundary": bool(unanimous_false_boundary[index]),
                       "repeated_false_zero": bool(repeated_false_zero[index])})
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol),
              "heldout_result_sha256": sha256(args.heldout_result),
              "correlations": correlations, "subsets": subsets,
              "diagnostic_markers": markers, "conclusion": conclusion,
              "groups": groups}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "correlations", "subsets", "diagnostic_markers", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
