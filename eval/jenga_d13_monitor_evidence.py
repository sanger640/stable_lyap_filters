"""Frozen D13 direct continuous monitor-evidence experiment."""
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

from action_branch_monitor import boundary_refinement_alarm
from consequence_monitor import (commitment_delta_bic, persistence_delta_bic,
                                 whole_trajectory_separation_curves)
from jenga_d6_set_response import sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_d12_model_ladder import load_records, response_normalizers
from jenga_reobservation_interface import compare_nested
from set_response_model import (DirectMonitorEvidenceModel,
                                GroupAmplitudeShapeTemporalCurveModel,
                                JointCurveMonitorEvidenceModel)


PROTOCOL = ROOT / "results/jenga/d13_monitor_evidence_protocol.json"
REPORT = ROOT / "results/jenga/d13_monitor_evidence_result.json"
TARGETS = ROOT / "results/jenga/d13_monitor_evidence_targets.npz"
OUTPUT = ROOT / "results/jenga/d13_monitor_evidence"
NAMES = ("early_boundary", "full_boundary", "commitment", "persistence")


def physical_evidence(actions, traces, response_scale):
    """Return unchanged v0 continuous evidence in the frozen four-value order."""
    rows = []
    for control, group in zip(actions, traces):
        normalized = group[..., :45] / np.maximum(response_scale[:45], 1e-6)
        difference = normalized[:, 0] - normalized[:, 1]
        early = np.sqrt(np.mean(difference[:, :18] ** 2, axis=(1, 2)))
        full = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))
        boundary = boundary_refinement_alarm(early[None], full[None])
        curve = whole_trajectory_separation_curves(group[-1:, :, :, :45])[0]
        action_difference = control[-1, 0, :8] - control[-1, 1, :8]
        commitment = commitment_delta_bic(curve, action_difference, 8, 5)[0]
        persistence = persistence_delta_bic(curve[8:], 10)
        rows.append((boundary.early_delta_bic[0], boundary.full_delta_bic[0],
                     commitment, persistence))
    values = np.asarray(rows, np.float32)
    if values.shape != (len(actions), 4) or not np.all(np.isfinite(values)):
        raise RuntimeError("physical evidence is malformed or non-finite")
    return values


def curve_evidence(actions, curve_values):
    """Recompute v0 evidence from predicted early/full gaps and separation curves."""
    rows = []
    for control, values in zip(actions, curve_values):
        boundary = boundary_refinement_alarm(values[None, :, 0], values[None, :, 1])
        curve = values[-1, 2:]
        action_difference = control[-1, 0, :8] - control[-1, 1, :8]
        commitment = commitment_delta_bic(curve, action_difference, 8, 5)[0]
        persistence = persistence_delta_bic(curve[8:], 10)
        rows.append((boundary.early_delta_bic[0], boundary.full_delta_bic[0],
                     commitment, persistence))
    values = np.asarray(rows, np.float32)
    if values.shape != (len(actions), 4) or not np.all(np.isfinite(values)):
        raise RuntimeError("curve-derived evidence is malformed or non-finite")
    return values


def signed_log(values):
    values = np.asarray(values, np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def target_normalizers(fit_raw):
    transformed = signed_log(fit_raw)
    mean = transformed.mean(0); scale = transformed.std(0)
    floor = max(float(np.sqrt(np.mean(scale ** 2))) * 1e-3, 1e-6)
    return mean.astype(np.float32), np.maximum(scale, floor).astype(np.float32)


def normalize_evidence(raw, mean, scale):
    return ((signed_log(raw) - mean) / scale).astype(np.float32)


def decode_evidence(normalized, mean, scale):
    transformed = np.asarray(normalized, np.float32) * scale + mean
    return (np.sign(transformed) * np.expm1(np.abs(transformed))).astype(np.float32)


def evidence_rows(raw):
    return [{"boundary": bool(row[0] > 0 and row[1] > 0),
             "commitment": bool(row[2] > 0), "persistence": bool(row[3] > 0),
             "alarm": bool(np.all(row > 0))} for row in np.asarray(raw)]


def evaluate_evidence(target_raw, predicted_raw, target_mean, target_scale):
    target_z = normalize_evidence(target_raw, target_mean, target_scale)
    predicted_z = normalize_evidence(predicted_raw, target_mean, target_scale)
    sign = np.sign(target_raw) == np.sign(predicted_raw)
    return {"groups": len(target_raw),
            "normalized_evidence_mse": float(np.mean((predicted_z - target_z) ** 2)),
            "transformed_evidence_mae": float(np.mean(np.abs(predicted_z - target_z))),
            "channel_sign_agreement": {name: float(sign[:, index].mean())
                                       for index, name in enumerate(NAMES)},
            "mean_channel_sign_agreement": float(sign.mean()),
            "nested": compare_nested(evidence_rows(target_raw), evidence_rows(predicted_raw))}


def predict_curve(model, data, norms, device):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(data[0], device=device),
                           torch.as_tensor(data[1], device=device), *norms[:4])[0]
    return decoded(prediction).cpu().numpy()


def predict_evidence(model, data, norms, target_mean, target_scale, device, joint=False):
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(data[0], device=device),
                       torch.as_tensor(data[1], device=device), *norms[:4])
        normalized = output[-1] if joint else output
    return decode_evidence(normalized.cpu().numpy(), target_mean, target_scale)


def invariance(model, data, norms, device, joint=False):
    count = min(2, len(data[0])); state = torch.as_tensor(data[0][:count], device=device)
    actions = torch.as_tensor(data[1][:count], device=device)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    model.eval()
    with torch.no_grad():
        original = model(state, actions, *norms[:4]); swapped = model(
            state, actions.flip(2), *norms[:4]); reordered = model(
            state, actions[:, permutation], *norms[:4])
    original = original[-1] if joint else original
    swapped = swapped[-1] if joint else swapped
    reordered = reordered[-1] if joint else reordered
    return {"endpoint_swap_max_abs": float(torch.max(torch.abs(original - swapped))),
            "level_permutation_max_abs": float(torch.max(torch.abs(original - reordered)))}


def train_arm(name, seed, protocol, fit, targets_z, curve_target, norms, device, output,
              protocol_hash):
    config = protocol["model"]; optimization = protocol["optimization"]
    torch.manual_seed(seed)
    common = dict(hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
                  action_horizon=config["action_horizon"], curve_steps=config["curve_steps"])
    model = (DirectMonitorEvidenceModel(**common) if name == "direct_evidence"
             else JointCurveMonitorEvidenceModel(**common)).to(device)
    start = torch.as_tensor(fit[0], device=device); actions = torch.as_tensor(fit[1], device=device)
    evidence_target = torch.as_tensor(targets_z, device=device)
    if name == "joint":
        target_curve = torch.as_tensor(curve_target, device=device)
        target_amplitude, target_shape, nonzero = factor_targets(target_curve)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    rng = np.random.default_rng([seed, 13]); history = []; began = time.time()
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(start), optimization["group_batch_size"],
                                            replace=False), device=device)
        model.train(); output_values = model(start[chosen], actions[chosen], *norms[:4])
        predicted_evidence = output_values[-1] if name == "joint" else output_values
        evidence_loss = (predicted_evidence - evidence_target[chosen]).square().mean()
        losses = {"evidence": evidence_loss}
        if name == "joint":
            _, amplitude, shape, _ = output_values
            losses.update(factor_losses(amplitude, shape, target_amplitude[chosen],
                                        target_shape[chosen], nonzero[chosen]))
        loss = sum(losses.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == optimization["steps"]:
            row = {"step": step, "total": float(loss.detach()),
                   **{key: float(value.detach()) for key, value in losses.items()}}
            history.append(row); print(f"D13 {name}: {row}", flush=True)
    checkpoint = output / f"{name}_seed{seed}.pt"
    torch.save({"model": model.state_dict(), "seed": seed, "arm": name,
                "protocol_sha256": protocol_hash}, checkpoint)
    return model, {"seed": seed, "seconds": time.time() - began, "history": history,
                   "checkpoint": str(checkpoint.relative_to(ROOT)),
                   "checkpoint_sha256": sha256(checkpoint)}


def gate(control, arm, config, symmetry):
    comparisons = {
        "normalized_evidence_mse_change": (arm["normalized_evidence_mse"]
                                            - control["normalized_evidence_mse"]),
        "mean_sign_agreement_change": (arm["mean_channel_sign_agreement"]
                                       - control["mean_channel_sign_agreement"]),
        "boundary_added_positive_reduction": (
            control["nested"]["boundary"]["added_positive_rate"]
            - arm["nested"]["boundary"]["added_positive_rate"]),
        "final_added_positive_reduction": (
            control["nested"]["alarm"]["added_positive_rate"]
            - arm["nested"]["alarm"]["added_positive_rate"]),
        "boundary_recall_change": (arm["nested"]["boundary"]["recall"]
                                   - control["nested"]["boundary"]["recall"]),
        "final_recall_change": (arm["nested"]["alarm"]["recall"]
                                - control["nested"]["alarm"]["recall"]),
        "symmetry_max_abs": max(symmetry.values())}
    tolerance = 1e-12
    checks = {
        "evidence_mse": comparisons["normalized_evidence_mse_change"] <= tolerance,
        "sign_agreement": comparisons["mean_sign_agreement_change"] >= -tolerance,
        "boundary_specificity": comparisons["boundary_added_positive_reduction"] >=
                                config["boundary_added_positive_reduction"] - tolerance,
        "final_specificity": comparisons["final_added_positive_reduction"] >=
                             config["final_added_positive_reduction"] - tolerance,
        "boundary_recall": comparisons["boundary_recall_change"] >=
                           -config["maximum_boundary_recall_loss"] - tolerance,
        "final_recall": comparisons["final_recall_change"] >=
                        -config["maximum_final_recall_loss"] - tolerance,
        "symmetry": comparisons["symmetry_max_abs"] <=
                    config["endpoint_and_level_symmetry_max_abs"]}
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--targets", default=str(TARGETS))
    parser.add_argument("--output", default=str(OUTPUT)); parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text()); protocol_hash = sha256(args.protocol)
    manifest_path = ROOT / protocol["data"]["manifest"]
    if sha256(manifest_path) != protocol["data"]["manifest_sha256"]: raise SystemExit("manifest hash differs")
    baseline_result = ROOT / protocol["frozen_curve_baseline"]["result"]
    baseline_checkpoint = ROOT / protocol["frozen_curve_baseline"]["checkpoint"]
    if sha256(baseline_result) != protocol["frozen_curve_baseline"]["result_sha256"]:
        raise SystemExit("baseline result hash differs")
    if sha256(baseline_checkpoint) != protocol["frozen_curve_baseline"]["checkpoint_sha256"]:
        raise SystemExit("baseline checkpoint hash differs")
    manifest = json.loads(manifest_path.read_text()); directories = {
        "d3": ROOT / protocol["data"]["d3_directory"],
        "d12": ROOT / protocol["data"]["d12_directory"]}
    folds = {fold: load_records(manifest, directories, {fold}) for fold in (
        "fit", "episode_validation", "configuration_validation", "joint_validation")}
    if any(len(folds[fold][0]) != count for fold, count in protocol["data"]["groups"].items()):
        raise SystemExit("fold counts differ from frozen protocol")
    normalizer_np = response_normalizers(*folds["fit"][:3]); response_scale = normalizer_np[4]
    raw_targets = {fold: physical_evidence(values[1], values[2], response_scale)
                   for fold, values in folds.items()}
    target_mean, target_scale = target_normalizers(raw_targets["fit"])
    normalized_targets = {fold: normalize_evidence(values, target_mean, target_scale)
                          for fold, values in raw_targets.items()}
    np.savez_compressed(args.targets, target_mean=target_mean, target_scale=target_scale,
                        **{f"{fold}_raw": values for fold, values in raw_targets.items()})
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    norms = tensors(normalizer_np, device); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = protocol["model"]
    baseline = GroupAmplitudeShapeTemporalCurveModel(
        hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        action_horizon=config["action_horizon"], curve_steps=config["curve_steps"]).to(device)
    saved = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
    baseline.load_state_dict(saved["model"])
    controls = {}
    for fold, values in folds.items():
        predicted_curves = predict_curve(baseline, values, norms, device)
        predicted_raw = curve_evidence(values[1], predicted_curves)
        controls[fold] = evaluate_evidence(raw_targets[fold], predicted_raw,
                                           target_mean, target_scale)
    curve_target = curve_targets(folds["fit"][2], response_scale)
    trained = {}; models = {}
    for arm in ("direct_evidence", "joint"):
        model, row = train_arm(
            arm, protocol["optimization"]["pilot_seed"], protocol, folds["fit"],
            normalized_targets["fit"], curve_target, norms, device, output, protocol_hash)
        models[arm] = model; trained[arm] = row
        row["invariance"] = invariance(model, folds["fit"], norms, device, joint=arm == "joint")
        row["evaluations"] = {}
        if arm == "joint": row["curve_derived_evaluations"] = {}
        for fold, values in folds.items():
            predicted_raw = predict_evidence(model, values, norms, target_mean, target_scale,
                                               device, joint=arm == "joint")
            row["evaluations"][fold] = evaluate_evidence(
                raw_targets[fold], predicted_raw, target_mean, target_scale)
            if arm == "joint":
                predicted_curve = predict_curve(model, values, norms, device)
                row["curve_derived_evaluations"][fold] = evaluate_evidence(
                    raw_targets[fold], curve_evidence(values[1], predicted_curve),
                    target_mean, target_scale)
    gates = {arm: {fold: gate(controls[fold], trained[arm]["evaluations"][fold],
                              protocol["gate_on_each_primary_axis"], trained[arm]["invariance"])
                   for fold in protocol["primary_validation_axes"]}
             for arm in trained}
    passing = [arm for arm in trained if all(gates[arm][fold]["passes"]
                                             for fold in protocol["primary_validation_axes"])]
    # Replication is intentionally a separate follow-up so this immutable pilot result cannot be
    # rewritten after seeing validation. The protocol merely authorizes it when the frozen gate passes.
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": protocol_hash, "device": device,
              "targets": {"path": str(Path(args.targets).relative_to(ROOT)),
                          "sha256": sha256(args.targets), "fit_mean": target_mean.tolist(),
                          "fit_scale": target_scale.tolist(),
                          "fit_positive_fraction": (raw_targets["fit"] > 0).mean(0).tolist()},
              "curve_baseline": {"evaluations": controls,
                                 "checkpoint": protocol["frozen_curve_baseline"]["checkpoint"]},
              "arms": trained, "gates": gates, "passing_arms": passing,
              "replication_authorized": bool(passing), "dev_or_test_read": False,
              "interpretation": ("pilot_pass_authorize_replication" if passing else
                                 "direct_evidence_failed_stop_privileged_jenga_encoder_family")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"curve_baseline": controls, "arms": {
        arm: row["evaluations"] for arm, row in trained.items()}, "gates": gates,
        "passing_arms": passing, "replication_authorized": bool(passing),
        "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
