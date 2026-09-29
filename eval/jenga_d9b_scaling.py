"""Fit-only episode-blocked D9b data-scaling experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
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
from jenga_d9_amplitude_shape import invariance
from jenga_d9b_heldout_train import zero_metrics
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_reobservation_interface import compare_nested, load_nested, nested_stage_rows, validation_rows
from set_response_model import GroupAmplitudeShapeTemporalCurveModel


PROTOCOL = ROOT / "results/jenga/d9b_scaling_protocol.json"
REPORT = ROOT / "results/jenga/d9b_scaling_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d9b_scaling"


def nested_episode_subsets(episodes, fold, fractions):
    ordered = sorted(map(str, episodes), key=lambda episode: hashlib.sha256(
        f"d9b-scaling-v1:{fold}:{episode}".encode()).hexdigest())
    return {float(fraction): ordered[:max(1, int(np.ceil(len(ordered) * fraction)))]
            for fraction in fractions}


def pilot_gate(aggregate, folds):
    keys = ("0.25", "0.5", "0.75", "1.0")
    errors = [aggregate[key]["mean_mse"] for key in keys]
    quarter = {row["fold"]: row["mean_mse"] for row in folds if row["fraction"] == .25}
    full = {row["fold"]: row["mean_mse"] for row in folds if row["fraction"] == 1.}
    improved_folds = sum(full[index] < quarter[index] for index in sorted(quarter))
    checks = {
        "monotone_mean_mse": all(right <= left for left, right in zip(errors, errors[1:])),
        "full_improves_20pct": errors[-1] <= .8 * errors[0],
        "four_of_five_folds_improve": improved_folds >= 4,
        "boundary_added_nonincreasing": (
            aggregate["1.0"]["nested"]["boundary"]["added_positive_rate"]
            <= aggregate["0.25"]["nested"]["boundary"]["added_positive_rate"]),
        "final_added_nonincreasing": (
            aggregate["1.0"]["nested"]["alarm"]["added_positive_rate"]
            <= aggregate["0.25"]["nested"]["alarm"]["added_positive_rate"]),
    }
    return {"checks": checks, "improved_folds": improved_folds,
            "full_vs_quarter_mse_ratio": errors[-1] / max(errors[0], 1e-12),
            "passes": all(checks.values())}


def evaluate(model, start_np, actions_np, truth_np, norms, response_scale, device):
    model.eval(); start = torch.as_tensor(start_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    target_amplitude, _, _ = factor_targets(target)
    with torch.no_grad():
        prediction, amplitude, _ = model(start, actions, *norms[:4])
        group_mse = torch.mean((prediction - target).square(), dim=(1, 2)).cpu().numpy()
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        predicted_values = decoded(prediction).cpu().numpy()
        predicted_amplitude = amplitude[:, 0, 0].cpu().numpy()
    reference = nested_stage_rows(actions_np, truth_np, response_scale)
    predicted = predicted_nested_rows(actions_np, predicted_values)
    return {"groups": len(start_np), "mean_mse": float(np.mean(group_mse)),
            "median_rmse": float(np.median(np.sqrt(group_mse))),
            "group_mse": group_mse.tolist(), "curve_errors": errors,
            "nested": compare_nested(reference, predicted),
            "reference_rows": reference, "predicted_rows": predicted,
            "target_amplitude": target_amplitude[:, 0, 0].cpu().tolist(),
            "predicted_amplitude": predicted_amplitude.tolist(),
            "zero": zero_metrics(target_amplitude[:, 0, 0].cpu().numpy(), predicted_amplitude),
            "invariance": invariance(model, start_np, actions_np, norms, device)}


def train_condition(seed, fold, fraction, protocol, train_data, validation_data, norms,
                    response_scale, device, output_dir):
    label = f"seed{seed}_fold{fold}_frac{int(round(100 * fraction)):03d}"
    cache = output_dir / f"{label}.json"; checkpoint = output_dir / f"{label}.pt"
    protocol_hash = sha256(PROTOCOL)
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("protocol_sha256") == protocol_hash:
            print(f"resume {label}", flush=True); return saved
    start_np, actions_np, truth_np, episodes = train_data
    target = torch.as_tensor(curve_targets(truth_np, response_scale), device=device)
    target_amplitude, target_shape, nonzero = factor_targets(target)
    start = torch.as_tensor(start_np, device=device); actions = torch.as_tensor(actions_np, device=device)
    torch.manual_seed(seed); rng = np.random.default_rng([seed, fold, int(round(fraction * 100))])
    model = GroupAmplitudeShapeTemporalCurveModel(
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"]).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history = []; began = time.time(); batch = optimization["group_batch_size"]
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(start), batch, replace=False), device=device)
        model.train(); _, amplitude, shape = model(start[chosen], actions[chosen], *norms[:4])
        losses = factor_losses(amplitude, shape, target_amplitude[chosen],
                               target_shape[chosen], nonzero[chosen])
        loss = sum(losses.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == optimization["steps"]:
            row = {"step": step, "total": float(loss.detach())}
            row.update({name: float(value.detach()) for name, value in losses.items()})
            history.append(row); print(f"{label}: {row}", flush=True)
    evaluation = evaluate(model, *validation_data, norms, response_scale, device)
    torch.save({"model": model.state_dict(), "seed": seed, "fold": fold,
                "fraction": fraction, "normalizers": [value.cpu() for value in norms],
                "protocol_sha256": protocol_hash}, checkpoint)
    result = {"protocol_sha256": protocol_hash, "seed": seed, "fold": fold,
              "fraction": fraction, "training_episodes": episodes,
              "training_groups": len(start_np), "seconds": time.time() - began,
              "history": history, "evaluation": evaluation,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint)}
    cache.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: {"training_groups": len(start_np),
                             "mean_mse": evaluation["mean_mse"],
                             "boundary": evaluation["nested"]["boundary"],
                             "alarm": evaluation["nested"]["alarm"]}}, indent=2), flush=True)
    return result


def aggregate_runs(rows, fractions):
    output = {}
    for fraction in fractions:
        selected = [row for row in rows if row["fraction"] == fraction]
        group_mse = np.concatenate([row["evaluation"]["group_mse"] for row in selected])
        reference = sum((row["evaluation"]["reference_rows"] for row in selected), [])
        predicted = sum((row["evaluation"]["predicted_rows"] for row in selected), [])
        target_amplitude = sum((row["evaluation"]["target_amplitude"] for row in selected), [])
        predicted_amplitude = sum((row["evaluation"]["predicted_amplitude"] for row in selected), [])
        output[str(fraction)] = {
            "runs": len(selected), "validation_groups": len(group_mse),
            "mean_training_groups": float(np.mean([row["training_groups"] for row in selected])),
            "mean_mse": float(np.mean(group_mse)),
            "median_rmse": float(np.median(np.sqrt(group_mse))),
            "nested": compare_nested(reference, predicted),
            "zero": zero_metrics(target_amplitude, predicted_amplitude),
        }
    if len(fractions) < 2:
        return output, {"log_error_vs_log_groups_slope": None, "intercept": None}
    x = np.log([output[str(value)]["mean_training_groups"] for value in fractions])
    y = np.log([output[str(value)]["mean_mse"] for value in fractions])
    slope, intercept = np.polyfit(x, y, 1)
    return output, {"log_error_vs_log_groups_slope": float(slope),
                    "intercept": float(intercept)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text())
    if sha256(args.data) != protocol["data"]["neighborhood_sha256"]:
        raise SystemExit("neighborhood data hash differs")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = np.load(args.data, allow_pickle=False)
    _, inspected = validation_rows(source["episode_id"]); inspected = list(map(str, inspected))
    if inspected != protocol["data"]["excluded_inspected_episode_ids"]:
        raise SystemExit("inspected episode split differs")
    eligible = [episode for episode in sorted(set(source["episode_id"].astype(str)), key=int)
                if episode not in set(inspected)]
    validation_folds = [eligible[index::protocol["data"]["folds"]]
                        for index in range(protocol["data"]["folds"])]
    if validation_folds != protocol["data"]["validation_episode_ids"]:
        raise SystemExit("validation folds differ")
    fractions = tuple(map(float, protocol["data"]["training_fractions"]))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    fold_data = []
    for fold, validation_episodes in enumerate(validation_folds):
        available = [episode for episode in eligible if episode not in set(validation_episodes)]
        subsets = nested_episode_subsets(available, fold, fractions)
        neighborhood_rows = np.flatnonzero(np.isin(source["episode_id"].astype(str), available))
        norm_np = normalizers(source["start"][neighborhood_rows], source["actions"][neighborhood_rows],
                              source["traces"][neighborhood_rows])
        validation = load_nested(args.nested, validation_episodes)[:3]
        fold_data.append((subsets, norm_np, validation))

    pilot = []
    for fold, (subsets, norm_np, validation) in enumerate(fold_data):
        norms = tensors(norm_np, device)
        for fraction in fractions:
            episodes = subsets[fraction]
            train_values = load_nested(args.nested, episodes)[:3]
            pilot.append(train_condition(
                protocol["optimization"]["pilot_seed"], fold, fraction, protocol,
                (*train_values, episodes), validation, norms, norm_np[4], device, output_dir))
    pilot_aggregate, pilot_slope = aggregate_runs(pilot, fractions)
    fold_rows = [{"fold": row["fold"], "fraction": row["fraction"],
                  "training_groups": row["training_groups"],
                  "mean_mse": row["evaluation"]["mean_mse"]} for row in pilot]
    gate = pilot_gate(pilot_aggregate, fold_rows)

    replication = []
    if gate["passes"]:
        replication_fractions = tuple(map(float, protocol["optimization"]["replication_fractions"]))
        for seed in protocol["optimization"]["replication_seeds_if_pilot_passes"]:
            for fold, (subsets, norm_np, validation) in enumerate(fold_data):
                norms = tensors(norm_np, device)
                for fraction in replication_fractions:
                    episodes = subsets[fraction]
                    train_values = load_nested(args.nested, episodes)[:3]
                    replication.append(train_condition(
                        seed, fold, fraction, protocol, (*train_values, episodes), validation,
                        norms, norm_np[4], device, output_dir))
        combined = pilot + replication
        replicated_aggregate, replicated_slope = aggregate_runs(combined, replication_fractions)
        ratio = (replicated_aggregate["1.0"]["mean_mse"]
                 / replicated_aggregate["0.25"]["mean_mse"])
        interpretation = ("replicated_clear_scaling_collect_generic_data" if ratio <= .8 else
                          "flat_scaling_test_generic_temporal_history" if ratio >= .9 else
                          "intermediate_run_data_by_history_factorial")
    else:
        replicated_aggregate = None; replicated_slope = None
        interpretation = "pilot_gate_failed_test_generic_temporal_history"

    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
              "pilot_runs": pilot, "pilot_aggregate": pilot_aggregate,
              "pilot_scaling": pilot_slope, "pilot_gate": gate,
              "replication_runs": replication,
              "replicated_aggregate": replicated_aggregate,
              "replicated_scaling": replicated_slope,
              "interpretation": interpretation}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pilot_aggregate": pilot_aggregate, "pilot_scaling": pilot_slope,
                      "pilot_gate": gate, "replication_runs": len(replication),
                      "replicated_aggregate": replicated_aggregate,
                      "replicated_scaling": replicated_slope,
                      "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
