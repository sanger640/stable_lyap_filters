"""Episode-cross-validated nonparametric response baseline for the D9b interface."""
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
from jenga_d7_explicit_curve import curve_targets, decoded, loss_terms, predicted_nested_rows
from jenga_d9b_heldout_train import zero_metrics
from jenga_d9b_support_audit import action_features, pairwise_rms
from jenga_reobservation_interface import compare_nested, load_nested, nested_stage_rows, validation_rows


PROTOCOL = ROOT / "results/jenga/d9b_knn_protocol.json"
REPORT = ROOT / "results/jenga/d9b_knn_result.json"


def episode_folds(episodes, count):
    episodes = np.asarray(episodes).astype(str)
    unique = sorted(set(episodes), key=int)
    return [np.flatnonzero(np.isin(episodes, unique[fold::count])) for fold in range(count)]


def knn_predict(distance, target, neighbors, kernel):
    distance = np.asarray(distance, float); target = np.asarray(target, float)
    order = np.argsort(distance, axis=1, kind="stable")[:, :neighbors]
    chosen_distance = np.take_along_axis(distance, order, axis=1)
    chosen_target = target[order]
    if kernel == "uniform":
        return chosen_target.mean(axis=1)
    if kernel != "inverse_distance": raise ValueError(f"unknown kernel {kernel}")
    exact = chosen_distance == 0
    weights = np.where(exact.any(axis=1, keepdims=True), exact,
                       1. / np.maximum(chosen_distance, 1e-12))
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(chosen_target * weights[..., None], axis=1)


def joint_distance(state_left, state_right, action_left, action_right, state_weight):
    state = pairwise_rms(state_left, state_right)
    action = pairwise_rms(action_left, action_right)
    return np.sqrt(state_weight * state ** 2 + (1. - state_weight) * action ** 2)


def interpretation(result, baseline):
    checks = {
        "curve_mse_improves_10pct": result["mean_mse"] <= .9 * baseline["mean_mse"],
        "one_monitor_agreement_not_worse": (
            result["boundary_agreement"] >= baseline["boundary_agreement"]
            or result["final_agreement"] >= baseline["final_agreement"]),
        "boundary_added_controlled": result["boundary_added"] <= baseline["boundary_added"] + .05,
        "final_added_controlled": result["final_added"] <= baseline["final_added"] + .05,
    }
    return {"checks": checks, "local_rule_beats_d9b": all(checks.values()),
            "decision": ("prioritize_neural_architecture_or_objective"
                         if all(checks.values()) else
                         "prioritize_feature_or_local_data_identifiability")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); output = Path(args.report)
    if output.exists() and not args.force: raise SystemExit(f"refusing to overwrite {output}")
    protocol = json.loads(Path(args.protocol).read_text())
    inputs = protocol["inputs"]
    for key, expected in ((args.data, inputs["neighborhood_sha256"]),
                          (ROOT / inputs["support_audit"], inputs["support_audit_sha256"]),
                          (ROOT / inputs["d9b_result"], inputs["d9b_result_sha256"])):
        if sha256(key) != expected: raise SystemExit(f"input hash differs: {key}")

    source = np.load(args.data, allow_pickle=False)
    val_rows, heldout_episodes = validation_rows(source["episode_id"])
    heldout_episodes = list(map(str, heldout_episodes))
    if heldout_episodes != inputs["heldout_episode_ids"]: raise SystemExit("held-out split differs")
    train_rows = np.setdiff1d(np.arange(len(source["start"])), val_rows)
    norm = normalizers(source["start"][train_rows], source["actions"][train_rows],
                       source["traces"][train_rows])
    all_episodes = sorted(set(source["episode_id"].astype(str)), key=int)
    fit_episodes = [episode for episode in all_episodes if episode not in set(heldout_episodes)]
    fit_start, fit_actions, fit_truth, fit_sources = load_nested(args.nested, fit_episodes)
    held_start, held_actions, held_truth, held_sources = load_nested(args.nested, heldout_episodes)
    if len(fit_start) != inputs["fit_groups"] or len(held_start) != inputs["heldout_groups"]:
        raise SystemExit("group counts differ")

    fit_state = (fit_start - norm[0]) / norm[1]; held_state = (held_start - norm[0]) / norm[1]
    fit_action = action_features(fit_actions, norm[2], norm[3])
    held_action = action_features(held_actions, norm[2], norm[3])
    fit_target = curve_targets(fit_truth, norm[4]).reshape(len(fit_truth), -1)
    held_target = curve_targets(held_truth, norm[4]).reshape(len(held_truth), -1)
    fit_episode_rows = np.asarray([name.split("_seed")[0][2:] for name in fit_sources])
    folds = episode_folds(fit_episode_rows, protocol["selection"]["folds"])

    candidates = []
    for state_weight in protocol["selection"]["state_weights"]:
        for neighbors in protocol["selection"]["neighbors"]:
            for kernel in protocol["selection"]["kernels"]:
                squared_errors = []; fold_rows = []
                for fold_index, validation in enumerate(folds):
                    training = np.setdiff1d(np.arange(len(fit_target)), validation)
                    distance = joint_distance(fit_state[validation], fit_state[training],
                                              fit_action[validation], fit_action[training], state_weight)
                    prediction = knn_predict(distance, fit_target[training], neighbors, kernel)
                    per_group = np.mean((prediction - fit_target[validation]) ** 2, axis=1)
                    squared_errors.extend(per_group.tolist())
                    fold_rows.append({"fold": fold_index, "groups": len(validation),
                                      "episodes": sorted(set(fit_episode_rows[validation]), key=int),
                                      "mean_mse": float(np.mean(per_group)),
                                      "median_rmse": float(np.median(np.sqrt(per_group)))})
                candidates.append({"state_weight": state_weight, "neighbors": neighbors,
                                   "kernel": kernel, "mean_mse": float(np.mean(squared_errors)),
                                   "median_rmse": float(np.median(np.sqrt(squared_errors))),
                                   "folds": fold_rows})
    selected_index = int(np.argmin([row["mean_mse"] for row in candidates]))
    selected = candidates[selected_index]

    distance = joint_distance(held_state, fit_state, held_action, fit_action,
                              selected["state_weight"])
    prediction = knn_predict(distance, fit_target, selected["neighbors"], selected["kernel"])
    group_mse = np.mean((prediction - held_target) ** 2, axis=1)
    prediction_3d = prediction.reshape(len(held_target), 6, 41)
    target_3d = held_target.reshape(len(held_target), 6, 41)
    errors = {name: float(value) for name, value in loss_terms(
        torch.as_tensor(prediction_3d), torch.as_tensor(target_3d)).items()}
    reference_rows = nested_stage_rows(held_actions, held_truth, norm[4])
    predicted_rows = predicted_nested_rows(held_actions, decoded(
        torch.as_tensor(prediction_3d)).numpy())
    nested = compare_nested(reference_rows, predicted_rows)
    target_amplitude = np.sqrt(np.mean(held_target ** 2, axis=1))
    predicted_amplitude = np.sqrt(np.mean(prediction ** 2, axis=1))
    zero = zero_metrics(target_amplitude, predicted_amplitude)

    audit = json.loads((ROOT / inputs["support_audit"]).read_text())
    flags = {row["index"]: (row["unanimous_false_boundary"], row["repeated_false_zero"])
             for row in audit["groups"]}
    evidence = []
    for index, source_name in enumerate(held_sources):
        order = np.argsort(distance[index], kind="stable")[:selected["neighbors"]]
        evidence.append({"index": index, "source": source_name,
                         "episode": source_name.split("_seed")[0][2:],
                         "group_rmse": float(np.sqrt(group_mse[index])),
                         "target_amplitude": float(target_amplitude[index]),
                         "predicted_amplitude": float(predicted_amplitude[index]),
                         "reference": reference_rows[index], "predicted": predicted_rows[index],
                         "neighbors": [{"source": fit_sources[row],
                                        "distance": float(distance[index, row])} for row in order],
                         "prior_unanimous_false_boundary": flags[index][0],
                         "prior_repeated_false_zero": flags[index][1]})

    measured = {"mean_mse": float(np.mean(group_mse)),
                "median_rmse": float(np.median(np.sqrt(group_mse))),
                "boundary_agreement": nested["boundary"]["agreement"],
                "boundary_added": nested["boundary"]["added_positive_rate"],
                "final_agreement": nested["alarm"]["agreement"],
                "final_added": nested["alarm"]["added_positive_rate"]}
    comparison = protocol["comparison"]
    baseline = {"mean_mse": comparison["d9b_mean_group_mse"],
                "boundary_agreement": comparison["d9b_median_boundary_agreement"],
                "boundary_added": comparison["d9b_median_boundary_added_positive_rate"],
                "final_agreement": comparison["d9b_median_final_agreement"],
                "final_added": comparison["d9b_median_final_added_positive_rate"]}
    decision = interpretation(measured, baseline)
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol), "selected_index": selected_index,
              "selected": selected, "candidate_ranking": sorted(
                  candidates, key=lambda row: row["mean_mse"]),
              "heldout": {"curve_errors": errors, "group_mean_mse": measured["mean_mse"],
                          "group_median_rmse": measured["median_rmse"], "nested": nested,
                          "zero": zero, "evidence": evidence},
              "d9b_baseline": baseline, "decision": decision}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": selected, "heldout": result["heldout"] | {"evidence": "omitted"},
                      "d9b_baseline": baseline, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
