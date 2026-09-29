"""Frozen D9c pilot: explicit generic interaction events versus D9b controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_set_response import DATA, NESTED, normalizers, sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded, loss_terms, predicted_nested_rows
from jenga_d9b_heldout_train import zero_metrics
from jenga_d9b_history_factorial import history_for_episodes
from jenga_d9b_scaling import aggregate_runs
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_reobservation_interface import compare_nested, load_nested, nested_stage_rows, validation_rows
from set_response_model import InteractionEventGroupAmplitudeShapeCurveModel


PROTOCOL = ROOT / "results/jenga/d9c_interaction_event_protocol.json"
REPORT = ROOT / "results/jenga/d9c_interaction_event_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d9c_interaction_event"


def event_normalizers(state_history, action_history):
    delta = state_history[:, 1:, :45] - state_history[:, :-1, :45]
    delta_mean = delta.mean(axis=(0, 1)); delta_scale = delta.std(axis=(0, 1))
    delta_floor = max(float(np.sqrt(np.mean(delta_scale ** 2))) * 1e-3, 1e-6)
    delta_scale = np.maximum(delta_scale, delta_floor)
    action_mean = action_history.mean(axis=(0, 1)); action_scale = action_history.std(axis=(0, 1))
    action_floor = max(float(np.sqrt(np.mean(action_scale ** 2))) * 1e-3, 1e-6)
    action_scale = np.maximum(action_scale, action_floor)
    return tuple(np.asarray(value, np.float32) for value in
                 (delta_mean, delta_scale, action_mean, action_scale))


def invariance(model, state_history, action_history, pair_actions, norms, device):
    count = min(2, len(state_history)); permutation = torch.tensor(
        [3, 0, 5, 1, 4, 2], device=device); inverse = torch.argsort(permutation)
    states = torch.as_tensor(state_history[:count], device=device)
    prior = torch.as_tensor(action_history[:count], device=device)
    controls = torch.as_tensor(pair_actions[:count], device=device)
    model.eval()
    with torch.no_grad():
        original = model(states, prior, controls, *norms)[0]
        swapped = model(states, prior, controls.flip(2), *norms)[0]
        reordered = model(states, prior, controls[:, permutation], *norms)[0][:, inverse]
    return {"endpoint_swap_max_abs": float(torch.max(torch.abs(original - swapped))),
            "level_permutation_max_abs": float(torch.max(torch.abs(original - reordered)))}


def evaluate(model, state_history_np, action_history_np, actions_np, truth_np,
             norms, response_scale, device):
    states = torch.as_tensor(state_history_np, device=device)
    prior = torch.as_tensor(action_history_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    target_amplitude, _, _ = factor_targets(target)
    model.eval()
    with torch.no_grad():
        prediction, amplitude, _ = model(states, prior, actions, *norms)
        group_mse = torch.mean((prediction - target).square(), dim=(1, 2)).cpu().numpy()
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        predicted_values = decoded(prediction).cpu().numpy()
        predicted_amplitude = amplitude[:, 0, 0].cpu().numpy()
    reference = nested_stage_rows(actions_np, truth_np, response_scale)
    predicted = predicted_nested_rows(actions_np, predicted_values)
    return {"groups": len(states), "mean_mse": float(np.mean(group_mse)),
            "median_rmse": float(np.median(np.sqrt(group_mse))),
            "group_mse": group_mse.tolist(), "curve_errors": errors,
            "nested": compare_nested(reference, predicted),
            "reference_rows": reference, "predicted_rows": predicted,
            "target_amplitude": target_amplitude[:, 0, 0].cpu().tolist(),
            "predicted_amplitude": predicted_amplitude.tolist(),
            "zero": zero_metrics(target_amplitude[:, 0, 0].cpu().numpy(), predicted_amplitude),
            "invariance": invariance(model, state_history_np, action_history_np,
                                     actions_np, norms, device)}


def train_condition(seed, fold, protocol, train_data, validation_data, norms,
                    response_scale, device, output_dir, protocol_hash):
    label = f"event_seed{seed}_fold{fold}_full"
    cache = output_dir / f"{label}.json"; checkpoint = output_dir / f"{label}.pt"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("protocol_sha256") == protocol_hash:
            print(f"resume {label}", flush=True); return saved
    state_history_np, action_history_np, actions_np, truth_np, episodes = train_data
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    target_amplitude, target_shape, nonzero = factor_targets(target)
    states = torch.as_tensor(state_history_np, device=device)
    prior = torch.as_tensor(action_history_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    torch.manual_seed(seed); rng = np.random.default_rng([seed, fold, 9, 3])
    model = InteractionEventGroupAmplitudeShapeCurveModel(
        history_states=protocol["data"]["history_states"],
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"], action_horizon=protocol["model"]["action_horizon"],
        curve_steps=protocol["model"]["curve_steps"]).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history_rows = []; began = time.time(); batch = optimization["group_batch_size"]
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(states), batch, replace=False), device=device)
        model.train(); _, amplitude, shape = model(states[chosen], prior[chosen], actions[chosen], *norms)
        losses = factor_losses(amplitude, shape, target_amplitude[chosen],
                               target_shape[chosen], nonzero[chosen])
        loss = sum(losses.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == optimization["steps"]:
            row = {"step": step, "total": float(loss.detach())}
            row.update({name: float(value.detach()) for name, value in losses.items()})
            history_rows.append(row); print(f"{label}: {row}", flush=True)
    evaluation = evaluate(model, *validation_data, norms, response_scale, device)
    torch.save({"model": model.state_dict(), "seed": seed, "fold": fold,
                "normalizers": [value.cpu() for value in norms],
                "protocol_sha256": protocol_hash}, checkpoint)
    result = {"protocol_sha256": protocol_hash, "arm": "interaction_event", "seed": seed,
              "fold": fold, "fraction": 1.0, "training_episodes": episodes,
              "training_groups": len(states), "seconds": time.time() - began,
              "history": history_rows, "evaluation": evaluation,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint)}
    cache.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {"mean_mse": evaluation["mean_mse"],
                             "boundary": evaluation["nested"]["boundary"],
                             "alarm": evaluation["nested"]["alarm"]}}, indent=2), flush=True)
    return result


def pilot_gate(snapshot, history, event, symmetry_max, config):
    comparisons = {
        "boundary_added_positive_reduction_from_snapshot": (
            snapshot["nested"]["boundary"]["added_positive_rate"]
            - event["nested"]["boundary"]["added_positive_rate"]),
        "final_added_positive_reduction_from_snapshot": (
            snapshot["nested"]["alarm"]["added_positive_rate"]
            - event["nested"]["alarm"]["added_positive_rate"]),
        "boundary_added_positive_change_from_history": (
            event["nested"]["boundary"]["added_positive_rate"]
            - history["nested"]["boundary"]["added_positive_rate"]),
        "final_added_positive_change_from_history": (
            event["nested"]["alarm"]["added_positive_rate"]
            - history["nested"]["alarm"]["added_positive_rate"]),
        "boundary_recall_change_from_snapshot": (
            event["nested"]["boundary"]["recall"] - snapshot["nested"]["boundary"]["recall"]),
        "final_recall_change_from_snapshot": (
            event["nested"]["alarm"]["recall"] - snapshot["nested"]["alarm"]["recall"]),
        "mean_mse_change_from_snapshot": event["mean_mse"] - snapshot["mean_mse"],
        "symmetry_max_abs": symmetry_max,
    }
    checks = {
        "boundary_specificity": comparisons["boundary_added_positive_reduction_from_snapshot"] >=
                                config["boundary_added_positive_reduction_from_snapshot"],
        "final_specificity": comparisons["final_added_positive_reduction_from_snapshot"] >=
                             config["final_added_positive_reduction_from_snapshot"],
        "no_worse_than_history": (comparisons["boundary_added_positive_change_from_history"] <= 0.
                                  and comparisons["final_added_positive_change_from_history"] <= 0.),
        "boundary_recall": comparisons["boundary_recall_change_from_snapshot"] >=
                           -config["maximum_boundary_recall_loss_from_snapshot"],
        "final_recall": comparisons["final_recall_change_from_snapshot"] >=
                        -config["maximum_final_recall_loss_from_snapshot"],
        "mean_mse": comparisons["mean_mse_change_from_snapshot"] <= 0.,
        "symmetry": symmetry_max <= config["endpoint_and_level_symmetry_max_abs"],
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
              (protocol["data"]["history_alignment_result"], protocol["data"]["history_alignment_result_sha256"]),
              (protocol["controls"]["snapshot_result"], protocol["controls"]["snapshot_result_sha256"]),
              (protocol["controls"]["history_result"], protocol["controls"]["history_result_sha256"]))
    for path, expected in frozen:
        resolved = Path(path) if Path(path).is_absolute() else ROOT / path
        if sha256(resolved) != expected: raise SystemExit(f"frozen input hash differs: {path}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = np.load(args.data, allow_pickle=False); history_data = np.load(args.history, allow_pickle=False)
    _, inspected = validation_rows(source["episode_id"]); inspected = list(map(str, inspected))
    if inspected != protocol["data"]["excluded_inspected_episode_ids"]: raise SystemExit("inspected split differs")
    eligible = [episode for episode in sorted(set(source["episode_id"].astype(str)), key=int)
                if episode not in set(inspected)]
    validation_folds = [eligible[index::protocol["data"]["folds"]]
                        for index in range(protocol["data"]["folds"])]
    if validation_folds != protocol["data"]["validation_episode_ids"]: raise SystemExit("folds differ")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    fold_data = []
    for validation_episodes in validation_folds:
        available = [episode for episode in eligible if episode not in set(validation_episodes)]
        neighborhood_rows = np.flatnonzero(np.isin(source["episode_id"].astype(str), available))
        response_norm = normalizers(source["start"][neighborhood_rows], source["actions"][neighborhood_rows], source["traces"][neighborhood_rows])
        available_history = history_for_episodes(history_data, available)
        event_norm = event_normalizers(*available_history)
        norm_np = (response_norm[0], response_norm[1], *event_norm,
                   response_norm[2], response_norm[3])
        train_nested = load_nested(args.nested, available)[:3]
        train_history = history_for_episodes(history_data, available, train_nested[0])
        validation_nested = load_nested(args.nested, validation_episodes)[:3]
        validation_history = history_for_episodes(history_data, validation_episodes, validation_nested[0])
        fold_data.append((available, norm_np, response_norm[4],
                          (*train_history, train_nested[1], train_nested[2]),
                          (*validation_history, validation_nested[1], validation_nested[2])))

    pilot = []
    for fold, (episodes, norm_np, response_scale, train_data, validation_data) in enumerate(fold_data):
        pilot.append(train_condition(protocol["optimization"]["pilot_seed"], fold, protocol,
                                     (*train_data, episodes), validation_data, tensors(norm_np, device),
                                     response_scale, device, output_dir, protocol_hash))
    event_aggregate, _ = aggregate_runs(pilot, (1.,))
    snapshot_result = json.loads((ROOT / protocol["controls"]["snapshot_result"]).read_text())
    snapshot_runs = [row for row in snapshot_result["pilot_runs"] if row["fraction"] == 1.]
    snapshot_aggregate, _ = aggregate_runs(snapshot_runs, (1.,))
    history_result = json.loads((ROOT / protocol["controls"]["history_result"]).read_text())
    history_aggregate = history_result["factorial"]["temporal"]["1.0"]
    symmetry_max = max(max(row["evaluation"]["invariance"].values()) for row in pilot)
    gate = pilot_gate(snapshot_aggregate["1.0"], history_aggregate, event_aggregate["1.0"],
                      symmetry_max, protocol["pilot_gate"])

    replication = []
    if gate["passes"]:
        for seed in protocol["optimization"]["replication_seeds_if_pilot_passes"]:
            for fold, (episodes, norm_np, response_scale, train_data, validation_data) in enumerate(fold_data):
                replication.append(train_condition(
                    seed, fold, protocol, (*train_data, episodes), validation_data,
                    tensors(norm_np, device), response_scale, device, output_dir, protocol_hash))
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": protocol_hash,
              "arms": {"snapshot": snapshot_aggregate["1.0"], "gru_history": history_aggregate,
                       "interaction_event": event_aggregate["1.0"]},
              "pilot_runs": pilot, "pilot_gate": gate, "replication_runs": replication,
              "interpretation": ("event_representation_passed_replicated" if gate["passes"]
                                 else "event_representation_failed_reconsider_curve_interface")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": result["arms"], "pilot_gate": gate,
                      "replication_runs": len(replication),
                      "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
