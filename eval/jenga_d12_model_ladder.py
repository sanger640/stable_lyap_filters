"""Run the frozen D12 deterministic, multimodal, and fixed-expert event-router ladder."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import normalizers, sha256, tensors
from jenga_d7_explicit_curve import curve_targets
from jenga_d9b_scaling import evaluate as deterministic_evaluate
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_d9c_interaction_event import event_normalizers
from jenga_d10_multimodal_curve import (evaluate as mixture_evaluate, invariance as mixture_invariance,
                                        mixture_nll)
from jenga_d11_fixed_expert_router import (assignment, event_features, evaluate_router,
                                           frozen_outputs)
from set_response_model import (EventCurveModeRouter, GroupAmplitudeShapeTemporalCurveModel,
                                MixtureGroupAmplitudeShapeCurveModel)


PROTOCOL = ROOT / "results/jenga/d12_ladder_protocol.json"
REPORT = ROOT / "results/jenga/d12_ladder_result.json"
OUTPUT = ROOT / "results/jenga/d12_ladder"


def load_records(manifest, directories, folds=None):
    selected = [row for row in manifest["records"] if folds is None or row["fold"] in folds]
    caches = {}; values = {name: [] for name in ("start", "actions", "traces")}
    metadata = []
    for row in selected:
        key = (row["dataset"], row["file"])
        if key not in caches:
            caches[key] = np.load(Path(directories[row["dataset"]]) / row["file"], allow_pickle=False)
        data = caches[key]; index = row["index"]
        for name in values:
            values[name].append(data[name][index])
        metadata.append(row)
    for data in caches.values():
        data.close()
    return (*(np.asarray(values[name], np.float32) for name in ("start", "actions", "traces")), metadata)


def history_lookup(path):
    data = np.load(path, allow_pickle=False)
    lookup = {(str(source), int(row)): (data["state_history"][index], data["action_history"][index])
              for index, (source, row) in enumerate(zip(data["source"], data["source_row"]))}
    data.close(); return lookup


def histories_for(metadata, lookups):
    values = [lookups[row["dataset"]][(Path(row["file"]).stem, row["index"])]
              for row in metadata]
    return np.asarray([row[0] for row in values], np.float32), np.asarray(
        [row[1] for row in values], np.float32)


def response_normalizers(start, actions, traces):
    return normalizers(start, actions.reshape(len(actions), -1, 38, 4),
                       traces.reshape(len(traces), -1, 38, 61))


def optimization(protocol, steps):
    values = protocol["optimization"]
    return {"steps": steps, "batch": values["group_batch_size"],
            "lr": values["initial_learning_rate"], "final_lr": values["final_learning_rate"],
            "clip": values["gradient_clip"]}


def train_deterministic(seed, protocol, train, validations, norms, response_scale, device,
                        output, protocol_hash):
    config = protocol["models"]; opt = optimization(protocol, protocol["optimization"]["curve_steps"])
    torch.manual_seed(seed)
    model = GroupAmplitudeShapeTemporalCurveModel(
        hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        action_horizon=config["action_horizon"], curve_steps=config["curve_steps"]).to(device)
    start_np, actions_np, truth_np, _ = train
    start = torch.as_tensor(start_np, device=device); actions = torch.as_tensor(actions_np, device=device)
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    amplitude_target, shape_target, nonzero = factor_targets(target)
    rng = np.random.default_rng([seed, 12, 9])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["steps"], eta_min=opt["final_lr"])
    history = []; began = time.time()
    for step in range(1, opt["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(start), opt["batch"], replace=False), device=device)
        model.train(); _, amplitude, shape = model(start[chosen], actions[chosen], *norms[:4])
        losses = factor_losses(amplitude, shape, amplitude_target[chosen], shape_target[chosen], nonzero[chosen])
        loss = sum(losses.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt["clip"]); optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == opt["steps"]:
            row = {"step": step, "total": float(loss.detach()),
                   **{name: float(value.detach()) for name, value in losses.items()}}
            history.append(row); print(f"D12 deterministic: {row}", flush=True)
    evaluations = {fold: deterministic_evaluate(model, *values[:3], norms, response_scale, device)
                   for fold, values in validations.items()}
    checkpoint = output / "deterministic_seed1.pt"
    torch.save({"model": model.state_dict(), "normalizers": [value.cpu() for value in norms],
                "seed": seed, "protocol_sha256": protocol_hash}, checkpoint)
    return model, {"seconds": time.time() - began, "history": history, "evaluations": evaluations,
                   "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}


def train_mixture(seed, protocol, train, validations, norms, response_scale, device,
                  output, protocol_hash):
    config = protocol["models"]; opt = optimization(protocol, protocol["optimization"]["curve_steps"])
    torch.manual_seed(seed)
    model = MixtureGroupAmplitudeShapeCurveModel(
        modes=config["modes"], hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        action_horizon=config["action_horizon"], curve_steps=config["curve_steps"]).to(device)
    start_np, actions_np, truth_np, _ = train
    start = torch.as_tensor(start_np, device=device); actions = torch.as_tensor(actions_np, device=device)
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    rng = np.random.default_rng([seed, 12, 10])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["steps"], eta_min=opt["final_lr"])
    history = []; began = time.time()
    for step in range(1, opt["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(start), opt["batch"], replace=False), device=device)
        model.train(); curves, logits, _ = model(start[chosen], actions[chosen], *norms[:4])
        loss = mixture_nll(curves, logits, target[chosen], config["fixed_sigma"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt["clip"]); optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == opt["steps"]:
            with torch.no_grad():
                oracle = torch.min(torch.mean((curves - target[chosen, None]).square(), dim=(2, 3)), dim=1).values.mean()
                probabilities = torch.softmax(logits, dim=-1).mean(0)
            row = {"step": step, "nll": float(loss.detach()), "batch_oracle_mse": float(oracle),
                   "mean_mode_probabilities": probabilities.cpu().tolist()}
            history.append(row); print(f"D12 mixture: {row}", flush=True)
    evaluations = {fold: mixture_evaluate(model, *values[:3], norms, response_scale, device)
                   for fold, values in validations.items()}
    symmetry = mixture_invariance(model, train[0], train[1], norms, device)
    checkpoint = output / "mixture_seed1.pt"
    torch.save({"model": model.state_dict(), "normalizers": [value.cpu() for value in norms],
                "seed": seed, "protocol_sha256": protocol_hash}, checkpoint)
    return model, {"seconds": time.time() - began, "history": history, "evaluations": evaluations,
                   "invariance": symmetry, "checkpoint": str(checkpoint.relative_to(ROOT)),
                   "checkpoint_sha256": sha256(checkpoint)}


def train_router(seed, protocol, mixture, train, validations, train_history, validation_history,
                 norms, response_scale, device, output, protocol_hash):
    config = protocol["models"]; opt = optimization(protocol, protocol["optimization"]["router_steps"])
    event_norm = event_normalizers(*train_history)
    train_events_np = event_features(*train_history, event_norm)
    train_features, train_curves, _, _ = frozen_outputs(
        mixture, train[0], train[1], norms, device)
    target = torch.as_tensor(curve_targets(train[2], response_scale), device=device)
    labels = assignment(train_curves, target)
    torch.manual_seed(seed)
    router = EventCurveModeRouter(
        event_dim=train_events_np.shape[-1], hidden=config["hidden"], modes=config["modes"]).to(device)
    events = torch.as_tensor(train_events_np, device=device)
    rng = np.random.default_rng([seed, 12, 11])
    optimizer = torch.optim.AdamW(router.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt["steps"], eta_min=opt["final_lr"])
    history = []; began = time.time()
    for step in range(1, opt["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(labels), opt["batch"], replace=False), device=device)
        router.train(); logits = router(train_features[chosen], events[chosen]); loss = F.cross_entropy(logits, labels[chosen])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), opt["clip"]); optimizer.step(); scheduler.step()
        if step == 1 or step % 500 == 0 or step == opt["steps"]:
            accuracy = float(torch.mean((logits.argmax(1) == labels[chosen]).float()).detach())
            row = {"step": step, "cross_entropy": float(loss.detach()), "batch_accuracy": accuracy}
            history.append(row); print(f"D12 event router: {row}", flush=True)
    router.eval()
    with torch.no_grad():
        training_accuracy = float(torch.mean((router(train_features, events).argmax(1) == labels).float()))
    evaluations = {}
    for fold, values in validations.items():
        states, prior = validation_history[fold]
        fold_events = torch.as_tensor(event_features(states, prior, event_norm), device=device)
        features, curves, _, amplitudes = frozen_outputs(mixture, values[0], values[1], norms, device)
        fold_target = torch.as_tensor(curve_targets(values[2], response_scale), device=device)
        evaluations[fold] = evaluate_router(
            router, "event_router", features, fold_events, curves, amplitudes, fold_target,
            values[0], values[1], values[2], response_scale, device)
    checkpoint = output / "event_router_seed1.pt"
    torch.save({"router": router.state_dict(), "event_normalizers": event_norm, "seed": seed,
                "protocol_sha256": protocol_hash}, checkpoint)
    return {"seconds": time.time() - began, "history": history,
            "training_assignment_counts": torch.bincount(labels, minlength=config["modes"]).cpu().tolist(),
            "training_accuracy": training_accuracy, "evaluations": evaluations,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}


def arm_gate(control, arm, config, oracle_mse=None, map_mse=None):
    comparisons = {
        "mean_mse_change": arm["mean_mse"] - control["mean_mse"],
        "boundary_added_positive_reduction": control["nested"]["boundary"]["added_positive_rate"] - arm["nested"]["boundary"]["added_positive_rate"],
        "final_added_positive_reduction": control["nested"]["alarm"]["added_positive_rate"] - arm["nested"]["alarm"]["added_positive_rate"],
        "boundary_recall_change": arm["nested"]["boundary"]["recall"] - control["nested"]["boundary"]["recall"],
        "final_recall_change": arm["nested"]["alarm"]["recall"] - control["nested"]["alarm"]["recall"]}
    tolerance = 1e-12
    checks = {"mean_mse": comparisons["mean_mse_change"] <= tolerance,
              "boundary_specificity": comparisons["boundary_added_positive_reduction"] >= config["boundary_added_positive_reduction"] - tolerance,
              "final_specificity": comparisons["final_added_positive_reduction"] >= config["final_added_positive_reduction"] - tolerance,
              "boundary_recall": comparisons["boundary_recall_change"] >= -config["maximum_boundary_recall_loss"] - tolerance,
              "final_recall": comparisons["final_recall_change"] >= -config["maximum_final_recall_loss"] - tolerance}
    if oracle_mse is not None:
        gap = max(map_mse - oracle_mse, 1e-12)
        comparisons["fraction_of_map_to_oracle_gap_closed"] = (map_mse - arm["mean_mse"]) / gap
        checks["oracle_gap"] = comparisons["fraction_of_map_to_oracle_gap_closed"] >= config[
            "router_minimum_fraction_of_map_to_oracle_mse_gap_closed"] - tolerance
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text()); protocol_hash = sha256(args.protocol)
    manifest_path = ROOT / protocol["data"]["manifest"]
    if sha256(manifest_path) != protocol["data"]["manifest_sha256"]: raise SystemExit("manifest hash differs")
    for key in ("d3", "d12"):
        if sha256(ROOT / protocol["data"][f"{key}_directory"] / "meta.json") != protocol["data"][f"{key}_meta_sha256"]:
            raise SystemExit(f"{key} metadata differs")
    history_report = json.loads((ROOT / "results/jenga/d12_history_alignment_result.json").read_text())
    d12_history_path = ROOT / protocol["data"]["d12_history"]
    if (history_report["protocol_sha256"] != protocol_hash or not history_report["gate"]["passes"]
            or history_report["output_sha256"] != sha256(d12_history_path)):
        raise SystemExit("D12 history is not aligned with this protocol")
    d3_history_path = ROOT / protocol["data"]["d3_history"]
    if sha256(d3_history_path) != protocol["data"]["d3_history_sha256"]: raise SystemExit("D3 history hash differs")
    manifest = json.loads(manifest_path.read_text()); directories = {
        "d3": ROOT / protocol["data"]["d3_directory"], "d12": ROOT / protocol["data"]["d12_directory"]}
    fit = load_records(manifest, directories, {"fit"})
    validations = {fold: load_records(manifest, directories, {fold}) for fold in (
        "episode_validation", "configuration_validation", "joint_validation")}
    if len(fit[0]) != protocol["data"]["groups"]["fit"] or any(
            len(values[0]) != protocol["data"]["groups"][fold] for fold, values in validations.items()):
        raise SystemExit("manifest group counts differ from frozen protocol")
    normalizer_np = response_normalizers(*fit[:3]); device = (
        "cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    norms = tensors(normalizer_np, device); response_scale = normalizer_np[4]
    lookups = {"d3": history_lookup(d3_history_path), "d12": history_lookup(d12_history_path)}
    fit_history = histories_for(fit[3], lookups)
    validation_history = {fold: histories_for(values[3], lookups) for fold, values in validations.items()}
    if not np.array_equal(fit_history[0][:, -1], fit[0]): raise SystemExit("fit histories do not align")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); seed = protocol["optimization"]["pilot_seed"]
    deterministic_model, deterministic = train_deterministic(
        seed, protocol, fit, validations, norms, response_scale, device, output, protocol_hash)
    del deterministic_model; torch.cuda.empty_cache() if device == "cuda" else None
    mixture_model, mixture = train_mixture(
        seed, protocol, fit, validations, norms, response_scale, device, output, protocol_hash)
    router = train_router(seed, protocol, mixture_model, fit, validations, fit_history,
                          validation_history, norms, response_scale, device, output, protocol_hash)
    gate_config = protocol["gate_on_each_primary_axis"]; gates = {"multimodal": {}, "event_router": {}}
    for fold in protocol["primary_validation_axes"]:
        control = deterministic["evaluations"][fold]; map_result = mixture["evaluations"][fold]
        oracle_mse = float(np.mean(map_result["oracle_best_group_mse"]))
        gates["multimodal"][fold] = arm_gate(control, map_result, gate_config)
        gates["event_router"][fold] = arm_gate(
            control, router["evaluations"][fold], gate_config, oracle_mse, map_result["mean_mse"])
    symmetry = max(mixture["invariance"].values())
    gates["multimodal"]["symmetry"] = symmetry <= gate_config["endpoint_and_level_symmetry_max_abs"]
    multimodal_pass = gates["multimodal"]["symmetry"] and all(
        gates["multimodal"][fold]["passes"] for fold in protocol["primary_validation_axes"])
    router_pass = all(gates["event_router"][fold]["passes"] for fold in protocol["primary_validation_axes"])
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": protocol_hash, "device": device,
              "groups": {"fit": len(fit[0]), **{fold: len(values[0]) for fold, values in validations.items()}},
              "normalizer_fit_groups_only": True, "deterministic": deterministic,
              "multimodal": mixture, "event_router": router, "gates": gates,
              "pilot_passes": {"multimodal": multimodal_pass, "event_router": router_pass},
              "replication_authorized": bool(multimodal_pass or router_pass),
              "dev_or_test_read": False,
              "interpretation": ("pilot_pass_authorize_replication" if multimodal_pass or router_pass
                                 else "pilot_failed_reconsider_privileged_model_monitor_interface")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    concise = {"groups": result["groups"], "gates": gates, "pilot_passes": result["pilot_passes"],
               "replication_authorized": result["replication_authorized"],
               "interpretation": result["interpretation"]}
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == "__main__":
    main()
