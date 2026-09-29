"""Frozen D11 pilot: task-free routers over fixed D10 response-curve experts."""
import argparse
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

from jenga_d6_set_response import DATA, NESTED, sha256
from jenga_d7_explicit_curve import curve_targets, decoded, loss_terms, predicted_nested_rows
from jenga_d9b_heldout_train import zero_metrics
from jenga_d9b_history_factorial import history_for_episodes
from jenga_d9b_scaling import aggregate_runs
from jenga_d9b_shared_amplitude import factor_targets
from jenga_d9c_interaction_event import event_normalizers
from jenga_reobservation_interface import compare_nested, load_nested, nested_stage_rows, validation_rows
from set_response_model import (EventCurveModeRouter, MixtureGroupAmplitudeShapeCurveModel,
                                SnapshotCurveModeRouter)


PROTOCOL = ROOT / "results/jenga/d11_fixed_expert_router_protocol.json"
REPORT = ROOT / "results/jenga/d11_fixed_expert_router_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d11_fixed_expert_router"


def event_features(state_history, action_history, normalizers):
    delta_mean, delta_scale, action_mean, action_scale = normalizers
    delta = state_history[:, 1:, :45] - state_history[:, :-1, :45]
    signed = (delta - delta_mean) / delta_scale
    magnitude = np.abs(delta) / delta_scale
    before = state_history[:, :-1, 45:57]; after = state_history[:, 1:, 45:57]
    created = np.maximum(after - before, 0.); lost = np.maximum(before - after, 0.)
    control = (action_history - action_mean) / action_scale
    return np.concatenate([signed, magnitude, before, after, created, lost, control], axis=-1).astype(np.float32)


def assignment(curves, target):
    """Target-derived component assignment, legal only inside a training fold or for diagnostics."""
    return torch.mean((curves - target[:, None]).square(), dim=(2, 3)).argmin(dim=1)


def frozen_outputs(model, start_np, actions_np, norms, device):
    start = torch.as_tensor(start_np, device=device); actions = torch.as_tensor(actions_np, device=device)
    model.eval()
    with torch.no_grad():
        features = model.encode_pairs(start, actions, *norms[:4])
        curves, logits, amplitudes = model(start, actions, *norms[:4])
    return features, curves, logits, amplitudes


def train_router(arm, seed, fold, protocol, pair_features, events, labels,
                 validation, response_scale, device, output_dir, protocol_hash):
    label = f"{arm}_seed{seed}_fold{fold}"
    cache = output_dir / f"{label}.json"; checkpoint = output_dir / f"{label}.pt"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("protocol_sha256") == protocol_hash:
            print(f"resume {label}", flush=True); return saved
    hidden = pair_features.shape[-1]; modes = protocol["frozen_model"]["modes"]
    torch.manual_seed(seed)
    router = (SnapshotCurveModeRouter(hidden=hidden, modes=modes) if arm == "snapshot_router"
              else EventCurveModeRouter(event_dim=events.shape[-1], hidden=hidden, modes=modes)).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(router.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    rng = np.random.default_rng([seed, fold, 11, int(arm == "event_router")])
    history = []; began = time.time(); batch = optimization["group_batch_size"]
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(labels), batch, replace=False), device=device)
        router.train()
        logits = (router(pair_features[chosen]) if arm == "snapshot_router"
                  else router(pair_features[chosen], events[chosen]))
        loss = F.cross_entropy(logits, labels[chosen])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 500 == 0 or step == optimization["steps"]:
            with torch.no_grad(): accuracy = torch.mean((logits.argmax(1) == labels[chosen]).float())
            row = {"step": step, "cross_entropy": float(loss.detach()),
                   "batch_accuracy": float(accuracy)}
            history.append(row); print(f"{label}: {row}", flush=True)
    evaluation = evaluate_router(router, arm, *validation, response_scale, device)
    router.eval()
    with torch.no_grad():
        train_logits = (router(pair_features) if arm == "snapshot_router"
                        else router(pair_features, events))
        training_accuracy = float(torch.mean((train_logits.argmax(1) == labels).float()))
    torch.save({"router": router.state_dict(), "arm": arm, "seed": seed, "fold": fold,
                "protocol_sha256": protocol_hash}, checkpoint)
    result = {"protocol_sha256": protocol_hash, "arm": arm, "seed": seed, "fold": fold,
              "fraction": 1.0, "training_groups": len(labels),
              "training_assignment_counts": torch.bincount(labels, minlength=modes).cpu().tolist(),
              "training_accuracy": training_accuracy, "seconds": time.time() - began,
              "history": history, "evaluation": evaluation,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint)}
    cache.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {"training_accuracy": training_accuracy,
                             "validation_routing_accuracy": evaluation["routing_accuracy"],
                             "mean_mse": evaluation["mean_mse"],
                             "boundary": evaluation["nested"]["boundary"],
                             "alarm": evaluation["nested"]["alarm"]}}, indent=2), flush=True)
    return result


def evaluate_router(router, arm, pair_features, events, curves, amplitudes, target,
                    start_np, actions_np, truth_np, response_scale, device):
    router.eval()
    with torch.no_grad():
        logits = (router(pair_features) if arm == "snapshot_router"
                  else router(pair_features, events))
        selected = logits.argmax(dim=1); rows = torch.arange(len(selected), device=device)
        prediction = curves[rows, selected]
        predicted_amplitude = amplitudes[rows, selected]
        oracle = assignment(curves, target)
        group_mse = torch.mean((prediction - target).square(), dim=(1, 2)).cpu().numpy()
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        predicted_values = decoded(prediction).cpu().numpy()
        target_amplitude, _, _ = factor_targets(target)
    reference = nested_stage_rows(actions_np, truth_np, response_scale)
    predicted = predicted_nested_rows(actions_np, predicted_values)
    modes = logits.shape[1]; confusion = np.zeros((modes, modes), dtype=int)
    np.add.at(confusion, (selected.cpu().numpy(), oracle.cpu().numpy()), 1)
    return {"groups": len(selected), "mean_mse": float(np.mean(group_mse)),
            "median_rmse": float(np.median(np.sqrt(group_mse))),
            "group_mse": group_mse.tolist(), "curve_errors": errors,
            "nested": compare_nested(reference, predicted),
            "reference_rows": reference, "predicted_rows": predicted,
            "target_amplitude": target_amplitude[:, 0, 0].cpu().tolist(),
            "predicted_amplitude": predicted_amplitude.cpu().tolist(),
            "zero": zero_metrics(target_amplitude[:, 0, 0].cpu().numpy(),
                                 predicted_amplitude.cpu().numpy()),
            "routing_accuracy": float(torch.mean((selected == oracle).float())),
            "selected_mode_counts": torch.bincount(selected, minlength=modes).cpu().tolist(),
            "oracle_mode_counts": torch.bincount(oracle, minlength=modes).cpu().tolist(),
            "selected_rows_oracle_columns_confusion": confusion.tolist()}


def gate(control, d10_map_mse, oracle_mse, routed, config):
    gap = d10_map_mse - oracle_mse
    closed = (d10_map_mse - routed["mean_mse"]) / max(gap, 1e-12)
    comparisons = {
        "fraction_of_map_to_oracle_gap_closed": closed,
        "boundary_added_positive_reduction": (
            control["nested"]["boundary"]["added_positive_rate"]
            - routed["nested"]["boundary"]["added_positive_rate"]),
        "final_added_positive_reduction": (
            control["nested"]["alarm"]["added_positive_rate"]
            - routed["nested"]["alarm"]["added_positive_rate"]),
        "boundary_recall_change": (routed["nested"]["boundary"]["recall"]
                                   - control["nested"]["boundary"]["recall"]),
        "final_recall_change": (routed["nested"]["alarm"]["recall"]
                                - control["nested"]["alarm"]["recall"]),
        "mean_mse_change": routed["mean_mse"] - control["mean_mse"],
    }
    checks = {
        "closes_oracle_gap": closed >= config["minimum_fraction_of_d10_map_to_oracle_mse_gap_closed"],
        "boundary_specificity": comparisons["boundary_added_positive_reduction"] >=
                                config["boundary_added_positive_reduction_from_d9b"],
        "final_specificity": comparisons["final_added_positive_reduction"] >=
                             config["final_added_positive_reduction_from_d9b"],
        "boundary_recall": comparisons["boundary_recall_change"] >=
                           -config["maximum_boundary_recall_loss_from_d9b"],
        "final_recall": comparisons["final_recall_change"] >=
                        -config["maximum_final_recall_loss_from_d9b"],
        "mean_mse": comparisons["mean_mse_change"] <= 0.,
    }
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--history", default=str(ROOT / "results/jenga/d9b_history_data.npz"))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR)); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text()); protocol_hash = sha256(args.protocol)
    frozen = ((args.data, protocol["data"]["neighborhood_sha256"]),
              (args.history, protocol["data"]["history_sha256"]),
              (protocol["data"]["d9b_control_result"],
               protocol["data"]["d9b_control_result_sha256"]),
              (protocol["frozen_experts"]["d10_result"], protocol["frozen_experts"]["d10_result_sha256"]))
    for path, expected in frozen:
        resolved = Path(path) if Path(path).is_absolute() else ROOT / path
        if sha256(resolved) != expected: raise SystemExit(f"frozen input hash differs: {path}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = np.load(args.data, allow_pickle=False); histories = np.load(args.history, allow_pickle=False)
    d10 = json.loads((ROOT / protocol["frozen_experts"]["d10_result"]).read_text())
    d10_protocol = d10["protocol"]; protocol["frozen_model"] = {
        key: d10_protocol["model"][key] for key in ("modes", "hidden", "heads", "layers", "action_horizon", "curve_steps")}
    _, inspected = validation_rows(source["episode_id"]); inspected = list(map(str, inspected))
    if inspected != protocol["data"]["excluded_inspected_episode_ids"]: raise SystemExit("inspected split differs")
    eligible = [episode for episode in sorted(set(source["episode_id"].astype(str)), key=int)
                if episode not in set(inspected)]
    validation_folds = [eligible[index::protocol["data"]["folds"]]
                        for index in range(protocol["data"]["folds"])]
    if validation_folds != protocol["data"]["validation_episode_ids"]: raise SystemExit("folds differ")
    d10_runs = {row["fold"]: row for row in d10["pilot_runs"]}
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for fold, validation_episodes in enumerate(validation_folds):
        available = [episode for episode in eligible if episode not in set(validation_episodes)]
        run = d10_runs[fold]; checkpoint_path = ROOT / run["checkpoint"]
        if sha256(checkpoint_path) != run["checkpoint_sha256"]: raise SystemExit("D10 checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = protocol["frozen_model"]
        model = MixtureGroupAmplitudeShapeCurveModel(**config).to(device)
        model.load_state_dict(checkpoint["model"]); norms = tuple(value.to(device) for value in checkpoint["normalizers"])
        response_scale = checkpoint["normalizers"][4].detach().cpu().numpy()

        train_nested = load_nested(args.nested, available)[:3]
        validation_nested = load_nested(args.nested, validation_episodes)[:3]
        train_history = history_for_episodes(histories, available, train_nested[0])
        validation_history = history_for_episodes(histories, validation_episodes, validation_nested[0])
        event_norm = event_normalizers(*train_history)
        train_events = torch.as_tensor(event_features(*train_history, event_norm), device=device)
        validation_events = torch.as_tensor(event_features(*validation_history, event_norm), device=device)
        train_features, train_curves, _, _ = frozen_outputs(model, train_nested[0], train_nested[1], norms, device)
        val_features, val_curves, _, val_amplitudes = frozen_outputs(model, validation_nested[0], validation_nested[1], norms, device)
        train_target = torch.as_tensor(curve_targets(train_nested[2], response_scale), device=device)
        val_target = torch.as_tensor(curve_targets(validation_nested[2], response_scale), device=device)
        labels = assignment(train_curves, train_target)
        validation = (val_features, validation_events, val_curves, val_amplitudes, val_target,
                      validation_nested[0], validation_nested[1], validation_nested[2])
        prepared.append((available, train_features, train_events, labels, validation, response_scale))

    pilot = []
    for fold, (episodes, features, events, labels, validation, response_scale) in enumerate(prepared):
        for arm in ("snapshot_router", "event_router"):
            pilot.append(train_router(arm, protocol["optimization"]["pilot_seed"], fold, protocol,
                                      features, events, labels, validation, response_scale,
                                      device, output_dir, protocol_hash))
    aggregates = {}
    gates = {}
    d9b = json.loads((ROOT / protocol["data"]["d9b_control_result"]).read_text())
    d9b_runs = [row for row in d9b["pilot_runs"] if row["fraction"] == 1.]
    d9b_aggregate, _ = aggregate_runs(d9b_runs, (1.,)); d9b_aggregate = d9b_aggregate["1.0"]
    d10_map_mse = d10["arms"]["multimodal_map"]["mean_mse"]
    oracle_mse = d10["distribution_diagnostic"]["oracle_best_mean_mse"]
    for arm in ("snapshot_router", "event_router"):
        selected = [row for row in pilot if row["arm"] == arm]
        aggregate, _ = aggregate_runs(selected, (1.,)); aggregates[arm] = aggregate["1.0"]
        aggregates[arm]["routing_accuracy"] = float(np.average(
            [row["evaluation"]["routing_accuracy"] for row in selected],
            weights=[row["evaluation"]["groups"] for row in selected]))
        aggregates[arm]["mean_training_accuracy"] = float(np.mean(
            [row["training_accuracy"] for row in selected]))
        gates[arm] = gate(d9b_aggregate, d10_map_mse, oracle_mse, aggregates[arm],
                          protocol["pilot_gate"])

    replication = []
    passing = [arm for arm in aggregates if gates[arm]["passes"]]
    for seed in protocol["optimization"]["replication_seeds_if_arm_passes"] if passing else []:
        for fold, (_, features, events, labels, validation, response_scale) in enumerate(prepared):
            for arm in passing:
                replication.append(train_router(arm, seed, fold, protocol, features, events, labels,
                                                 validation, response_scale, device, output_dir,
                                                 protocol_hash))
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": protocol_hash,
              "controls": {"d9b": d9b_aggregate, "d10_map_mse": d10_map_mse,
                           "d10_oracle_best_mse": oracle_mse},
              "arms": aggregates, "pilot_runs": pilot, "pilot_gates": gates,
              "replication_runs": replication,
              "interpretation": ("router_passed_replicated" if passing else
                                 "routers_failed_collect_broader_generic_data")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": aggregates, "pilot_gates": gates,
                      "replication_runs": len(replication),
                      "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
