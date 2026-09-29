"""Frozen D10 pilot: a task-free categorical mixture over complete response curves."""
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
from jenga_d9b_scaling import aggregate_runs
from jenga_d9b_shared_amplitude import factor_targets
from jenga_reobservation_interface import compare_nested, load_nested, nested_stage_rows, validation_rows
from set_response_model import MixtureGroupAmplitudeShapeCurveModel


PROTOCOL = ROOT / "results/jenga/d10_multimodal_curve_protocol.json"
REPORT = ROOT / "results/jenga/d10_multimodal_curve_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d10_multimodal_curve"


def mixture_nll(curves, logits, target, sigma):
    """Negative log likelihood under a fixed-variance categorical Gaussian mixture."""
    if curves.ndim != 4 or target.shape != curves.shape[:1] + curves.shape[2:]:
        raise ValueError("curves/target must have shapes (B,K,L,D)/(B,L,D)")
    squared_error = (curves - target[:, None]).square().sum(dim=(2, 3))
    component_log_prob = -squared_error / (2. * sigma ** 2)
    return -torch.logsumexp(torch.log_softmax(logits, dim=-1) + component_log_prob,
                            dim=-1).mean()


def invariance(model, start, actions, norms, device):
    count = min(2, len(start)); permutation = torch.tensor(
        [3, 0, 5, 1, 4, 2], device=device); inverse = torch.argsort(permutation)
    state = torch.as_tensor(start[:count], device=device)
    control = torch.as_tensor(actions[:count], device=device)
    model.eval()
    with torch.no_grad():
        original = model(state, control, *norms[:4])
        swapped = model(state, control.flip(2), *norms[:4])
        reordered = model(state, control[:, permutation], *norms[:4])
    return {"endpoint_swap_curve_max_abs": float(torch.max(torch.abs(original[0] - swapped[0]))),
            "endpoint_swap_logit_max_abs": float(torch.max(torch.abs(original[1] - swapped[1]))),
            "level_permutation_curve_max_abs": float(torch.max(torch.abs(
                original[0] - reordered[0][:, :, inverse]))),
            "level_permutation_logit_max_abs": float(torch.max(torch.abs(
                original[1] - reordered[1])))}


def evaluate(model, start_np, actions_np, truth_np, norms, response_scale, device):
    start = torch.as_tensor(start_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    target_amplitude, _, _ = factor_targets(target)
    model.eval()
    with torch.no_grad():
        curves, logits, amplitude = model(start, actions, *norms[:4])
        probabilities = torch.softmax(logits, dim=-1)
        map_modes = logits.argmax(dim=-1); rows = torch.arange(len(start), device=device)
        map_prediction = curves[rows, map_modes]
        per_mode_mse = torch.mean((curves - target[:, None]).square(), dim=(2, 3))
        oracle_modes = per_mode_mse.argmin(dim=-1)
        oracle_prediction = curves[rows, oracle_modes]
        mixture_mean = torch.sum(probabilities[:, :, None, None] * curves, dim=1)
        map_group_mse = torch.mean((map_prediction - target).square(), dim=(1, 2)).cpu().numpy()
        oracle_group_mse = torch.mean((oracle_prediction - target).square(), dim=(1, 2)).cpu().numpy()
        mean_group_mse = torch.mean((mixture_mean - target).square(), dim=(1, 2)).cpu().numpy()
        errors = {name: float(value) for name, value in loss_terms(map_prediction, target).items()}
        predicted_values = decoded(map_prediction).cpu().numpy()
        predicted_amplitude = amplitude[rows, map_modes].cpu().numpy()
    reference = nested_stage_rows(actions_np, truth_np, response_scale)
    predicted = predicted_nested_rows(actions_np, predicted_values)
    return {"groups": len(start), "mean_mse": float(np.mean(map_group_mse)),
            "median_rmse": float(np.median(np.sqrt(map_group_mse))),
            "group_mse": map_group_mse.tolist(), "curve_errors": errors,
            "nested": compare_nested(reference, predicted),
            "reference_rows": reference, "predicted_rows": predicted,
            "target_amplitude": target_amplitude[:, 0, 0].cpu().tolist(),
            "predicted_amplitude": predicted_amplitude.tolist(),
            "zero": zero_metrics(target_amplitude[:, 0, 0].cpu().numpy(), predicted_amplitude),
            "invariance": invariance(model, start_np, actions_np, norms, device),
            "oracle_best_group_mse": oracle_group_mse.tolist(),
            "mixture_mean_group_mse": mean_group_mse.tolist(),
            "map_modes": map_modes.cpu().tolist(), "oracle_modes": oracle_modes.cpu().tolist(),
            "mean_mode_probabilities": probabilities.mean(dim=0).cpu().tolist()}


def train_condition(seed, fold, protocol, train_data, validation_data, norms,
                    response_scale, device, output_dir, protocol_hash):
    label = f"mixture_seed{seed}_fold{fold}_full"
    cache = output_dir / f"{label}.json"; checkpoint = output_dir / f"{label}.pt"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("protocol_sha256") == protocol_hash:
            print(f"resume {label}", flush=True); return saved
    start_np, actions_np, truth_np, episodes = train_data
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    start = torch.as_tensor(start_np, device=device); actions = torch.as_tensor(actions_np, device=device)
    torch.manual_seed(seed); rng = np.random.default_rng([seed, fold, 10])
    model = MixtureGroupAmplitudeShapeCurveModel(
        modes=protocol["model"]["modes"], hidden=protocol["model"]["hidden"],
        heads=protocol["model"]["heads"], layers=protocol["model"]["layers"],
        action_horizon=protocol["model"]["action_horizon"],
        curve_steps=protocol["model"]["curve_steps"]).to(device)
    optimization = protocol["optimization"]; sigma = protocol["model"]["fixed_sigma"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history = []; began = time.time(); batch = optimization["group_batch_size"]
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(start), batch, replace=False), device=device)
        model.train(); curves, logits, _ = model(start[chosen], actions[chosen], *norms[:4])
        loss = mixture_nll(curves, logits, target[chosen], sigma)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == optimization["steps"]:
            with torch.no_grad():
                probabilities = torch.softmax(logits, dim=-1).mean(dim=0)
                min_mse = torch.mean(torch.min(torch.mean(
                    (curves - target[chosen, None]).square(), dim=(2, 3)), dim=1).values)
            row = {"step": step, "nll": float(loss.detach()),
                   "batch_oracle_mse": float(min_mse),
                   "mean_mode_probabilities": probabilities.cpu().tolist()}
            history.append(row); print(f"{label}: {row}", flush=True)
    evaluation = evaluate(model, *validation_data, norms, response_scale, device)
    torch.save({"model": model.state_dict(), "seed": seed, "fold": fold,
                "normalizers": [value.cpu() for value in norms],
                "protocol_sha256": protocol_hash}, checkpoint)
    result = {"protocol_sha256": protocol_hash, "arm": "multimodal_curve", "seed": seed,
              "fold": fold, "fraction": 1.0, "training_episodes": episodes,
              "training_groups": len(start), "seconds": time.time() - began,
              "history": history, "evaluation": evaluation,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint)}
    cache.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {"mean_mse": evaluation["mean_mse"],
                             "oracle_best_mse": float(np.mean(evaluation["oracle_best_group_mse"])),
                             "mode_probabilities": evaluation["mean_mode_probabilities"],
                             "boundary": evaluation["nested"]["boundary"],
                             "alarm": evaluation["nested"]["alarm"]}}, indent=2), flush=True)
    return result


def diagnostic_aggregate(rows, modes):
    oracle = np.concatenate([row["evaluation"]["oracle_best_group_mse"] for row in rows])
    mean = np.concatenate([row["evaluation"]["mixture_mean_group_mse"] for row in rows])
    selected = np.concatenate([row["evaluation"]["map_modes"] for row in rows])
    oracle_selected = np.concatenate([row["evaluation"]["oracle_modes"] for row in rows])
    confusion = np.zeros((modes, modes), dtype=int)
    np.add.at(confusion, (selected, oracle_selected), 1)
    return {"oracle_best_mean_mse": float(np.mean(oracle)),
            "mixture_mean_mse": float(np.mean(mean)),
            "map_mode_counts": np.bincount(selected, minlength=modes).tolist(),
            "oracle_mode_counts": np.bincount(oracle_selected, minlength=modes).tolist(),
            "map_vs_oracle_mode_accuracy": float(np.mean(selected == oracle_selected)),
            "map_rows_oracle_columns_confusion": confusion.tolist(),
            "oracle_best_improvement_over_map": None}


def pilot_gate(snapshot, mixture, symmetry_max, config):
    comparisons = {
        "boundary_added_positive_reduction": (
            snapshot["nested"]["boundary"]["added_positive_rate"]
            - mixture["nested"]["boundary"]["added_positive_rate"]),
        "final_added_positive_reduction": (
            snapshot["nested"]["alarm"]["added_positive_rate"]
            - mixture["nested"]["alarm"]["added_positive_rate"]),
        "boundary_recall_change": (mixture["nested"]["boundary"]["recall"]
                                   - snapshot["nested"]["boundary"]["recall"]),
        "final_recall_change": (mixture["nested"]["alarm"]["recall"]
                                - snapshot["nested"]["alarm"]["recall"]),
        "mean_mse_change": mixture["mean_mse"] - snapshot["mean_mse"],
        "symmetry_max_abs": symmetry_max,
    }
    checks = {
        "boundary_specificity": comparisons["boundary_added_positive_reduction"] >=
                                config["boundary_added_positive_reduction_from_snapshot"],
        "final_specificity": comparisons["final_added_positive_reduction"] >=
                             config["final_added_positive_reduction_from_snapshot"],
        "boundary_recall": comparisons["boundary_recall_change"] >=
                           -config["maximum_boundary_recall_loss_from_snapshot"],
        "final_recall": comparisons["final_recall_change"] >=
                        -config["maximum_final_recall_loss_from_snapshot"],
        "mean_mse": comparisons["mean_mse_change"] <= 0.,
        "symmetry": symmetry_max <= config["endpoint_and_level_symmetry_max_abs"],
    }
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR)); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text()); protocol_hash = sha256(args.protocol)
    for path, expected in ((args.data, protocol["data"]["neighborhood_sha256"]),
                           (protocol["control"]["snapshot_result"], protocol["control"]["snapshot_result_sha256"])):
        resolved = Path(path) if Path(path).is_absolute() else ROOT / path
        if sha256(resolved) != expected: raise SystemExit(f"frozen input hash differs: {path}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = np.load(args.data, allow_pickle=False)
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
        norm_np = normalizers(source["start"][neighborhood_rows], source["actions"][neighborhood_rows], source["traces"][neighborhood_rows])
        fold_data.append((available, norm_np, load_nested(args.nested, available)[:3],
                          load_nested(args.nested, validation_episodes)[:3]))

    pilot = []
    for fold, (episodes, norm_np, train_data, validation_data) in enumerate(fold_data):
        pilot.append(train_condition(protocol["optimization"]["pilot_seed"], fold, protocol,
                                     (*train_data, episodes), validation_data, tensors(norm_np, device),
                                     norm_np[4], device, output_dir, protocol_hash))
    mixture_aggregate, _ = aggregate_runs(pilot, (1.,)); mixture_aggregate = mixture_aggregate["1.0"]
    diagnostic = diagnostic_aggregate(pilot, protocol["model"]["modes"])
    diagnostic["oracle_best_improvement_over_map"] = (
        1. - diagnostic["oracle_best_mean_mse"] / mixture_aggregate["mean_mse"])
    snapshot_result = json.loads((ROOT / protocol["control"]["snapshot_result"]).read_text())
    snapshot_runs = [row for row in snapshot_result["pilot_runs"] if row["fraction"] == 1.]
    snapshot_aggregate, _ = aggregate_runs(snapshot_runs, (1.,)); snapshot_aggregate = snapshot_aggregate["1.0"]
    symmetry_max = max(max(row["evaluation"]["invariance"].values()) for row in pilot)
    gate = pilot_gate(snapshot_aggregate, mixture_aggregate, symmetry_max, protocol["pilot_gate"])

    replication = []
    if gate["passes"]:
        for seed in protocol["optimization"]["replication_seeds_if_pilot_passes"]:
            for fold, (episodes, norm_np, train_data, validation_data) in enumerate(fold_data):
                replication.append(train_condition(
                    seed, fold, protocol, (*train_data, episodes), validation_data,
                    tensors(norm_np, device), norm_np[4], device, output_dir, protocol_hash))
    if gate["passes"]:
        interpretation = "multimodal_curve_passed_replicated"
    elif diagnostic["oracle_best_improvement_over_map"] >= .10:
        interpretation = "mode_inference_bottleneck"
    else:
        interpretation = "no_multimodal_coverage_gain_stop_architecture_iteration"
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": protocol_hash,
              "arms": {"snapshot": snapshot_aggregate, "multimodal_map": mixture_aggregate},
              "distribution_diagnostic": diagnostic, "pilot_runs": pilot,
              "pilot_gate": gate, "replication_runs": replication, "interpretation": interpretation}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": result["arms"], "distribution_diagnostic": diagnostic,
                      "pilot_gate": gate, "replication_runs": len(replication),
                      "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
