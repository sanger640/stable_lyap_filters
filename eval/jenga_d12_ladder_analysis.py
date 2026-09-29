"""Read-only fit/stratum analysis of the completed frozen D12 ladder."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded, predicted_nested_rows
from jenga_d9b_scaling import evaluate as deterministic_evaluate
from jenga_d9c_interaction_event import event_normalizers
from jenga_d10_multimodal_curve import evaluate as mixture_evaluate
from jenga_d11_fixed_expert_router import event_features, evaluate_router, frozen_outputs
from jenga_d12_model_ladder import (histories_for, history_lookup, load_records,
                                    response_normalizers)
from jenga_reobservation_interface import compare_nested, nested_stage_rows
from set_response_model import (EventCurveModeRouter, GroupAmplitudeShapeTemporalCurveModel,
                                MixtureGroupAmplitudeShapeCurveModel)


RESULT = ROOT / "results/jenga/d12_ladder_result.json"
OUTPUT = ROOT / "results/jenga/d12_ladder_analysis.json"


def stratum_summary(metadata, evaluation):
    values = defaultdict(lambda: {"groups": 0, "mse": [], "boundary_added": 0,
                                  "boundary_negative": 0, "final_added": 0, "final_negative": 0})
    for row, mse, reference, predicted in zip(metadata, evaluation["group_mse"],
                                               evaluation["reference_rows"],
                                               evaluation["predicted_rows"]):
        name = row.get("event_stratum", "d3_prior"); item = values[name]
        item["groups"] += 1; item["mse"].append(mse)
        if not reference["boundary"]:
            item["boundary_negative"] += 1; item["boundary_added"] += int(predicted["boundary"])
        if not reference["alarm"]:
            item["final_negative"] += 1; item["final_added"] += int(predicted["alarm"])
    return {name: {**{key: value for key, value in item.items() if key != "mse"},
                   "mean_mse": float(np.mean(item["mse"]))}
            for name, item in sorted(values.items())}


def mse_oracle_monitor(model, data, norms, response_scale, device):
    start, actions, truth = data[:3]
    with torch.no_grad():
        curves, _, _ = model(torch.as_tensor(start, device=device),
                             torch.as_tensor(actions, device=device), *norms[:4])
        target = torch.as_tensor(curve_targets(truth, response_scale), device=device)
        modes = torch.mean((curves - target[:, None]).square(), dim=(2, 3)).argmin(dim=1)
        prediction = curves[torch.arange(len(start), device=device), modes]
    reference = nested_stage_rows(actions, truth, response_scale)
    predicted = predicted_nested_rows(actions, decoded(prediction).cpu().numpy())
    return {"nested": compare_nested(reference, predicted), "diagnostic_only": True}


def main():
    result = json.loads(RESULT.read_text()); protocol = result["protocol"]
    manifest = json.loads((ROOT / protocol["data"]["manifest"]).read_text())
    directories = {name: ROOT / protocol["data"][f"{name}_directory"] for name in ("d3", "d12")}
    datasets = {fold: load_records(manifest, directories, {fold}) for fold in (
        "fit", "episode_validation", "configuration_validation", "joint_validation")}
    fit = datasets["fit"]; normalizer_np = response_normalizers(*fit[:3])
    device = "cuda" if torch.cuda.is_available() else "cpu"; norms = tensors(normalizer_np, device)
    config = protocol["models"]

    deterministic = GroupAmplitudeShapeTemporalCurveModel(
        hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        action_horizon=config["action_horizon"], curve_steps=config["curve_steps"]).to(device)
    det_checkpoint = torch.load(ROOT / result["deterministic"]["checkpoint"], map_location=device,
                                weights_only=False)
    if sha256(ROOT / result["deterministic"]["checkpoint"]) != result["deterministic"]["checkpoint_sha256"]:
        raise SystemExit("deterministic checkpoint hash differs")
    deterministic.load_state_dict(det_checkpoint["model"])

    mixture = MixtureGroupAmplitudeShapeCurveModel(
        modes=config["modes"], hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        action_horizon=config["action_horizon"], curve_steps=config["curve_steps"]).to(device)
    mix_checkpoint = torch.load(ROOT / result["multimodal"]["checkpoint"], map_location=device,
                                weights_only=False)
    if sha256(ROOT / result["multimodal"]["checkpoint"]) != result["multimodal"]["checkpoint_sha256"]:
        raise SystemExit("mixture checkpoint hash differs")
    mixture.load_state_dict(mix_checkpoint["model"])

    lookups = {"d3": history_lookup(ROOT / protocol["data"]["d3_history"]),
               "d12": history_lookup(ROOT / protocol["data"]["d12_history"])}
    histories = {fold: histories_for(values[3], lookups) for fold, values in datasets.items()}
    router_checkpoint = torch.load(ROOT / result["event_router"]["checkpoint"], map_location=device,
                                   weights_only=False)
    if sha256(ROOT / result["event_router"]["checkpoint"]) != result["event_router"]["checkpoint_sha256"]:
        raise SystemExit("router checkpoint hash differs")
    event_norm = router_checkpoint["event_normalizers"]
    router = EventCurveModeRouter(event_dim=142, hidden=config["hidden"], modes=config["modes"]).to(device)
    router.load_state_dict(router_checkpoint["router"])

    evaluations = {"deterministic": {}, "multimodal": {}, "event_router": {}}
    strata = {name: {} for name in evaluations}; oracle_monitor = {}
    for fold, values in datasets.items():
        evaluations["deterministic"][fold] = deterministic_evaluate(
            deterministic, *values[:3], norms, normalizer_np[4], device)
        evaluations["multimodal"][fold] = mixture_evaluate(
            mixture, *values[:3], norms, normalizer_np[4], device)
        oracle_monitor[fold] = mse_oracle_monitor(mixture, values, norms, normalizer_np[4], device)
        features, curves, _, amplitudes = frozen_outputs(mixture, values[0], values[1], norms, device)
        states, prior = histories[fold]
        events = torch.as_tensor(event_features(states, prior, event_norm), device=device)
        target = torch.as_tensor(curve_targets(values[2], normalizer_np[4]), device=device)
        evaluations["event_router"][fold] = evaluate_router(
            router, "event_router", features, events, curves, amplitudes, target,
            values[0], values[1], values[2], normalizer_np[4], device)
        for arm in evaluations:
            strata[arm][fold] = stratum_summary(values[3], evaluations[arm][fold])
    concise = {arm: {fold: {"mean_mse": values["mean_mse"], "nested": values["nested"],
                             **({"routing_accuracy": values["routing_accuracy"]}
                                if "routing_accuracy" in values else {})}
                          for fold, values in folds.items()}
               for arm, folds in evaluations.items()}
    output = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "result_sha256": sha256(RESULT), "concise": concise,
              "mse_oracle_monitor": oracle_monitor, "stratum_diagnostics": strata,
              "selection_labels_used": False}
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n"); print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
