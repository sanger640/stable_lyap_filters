"""D20 stages 2-4: H8 rollout, local response geometry, then frozen monitor agreement."""
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

from enhanced_hybrid_dynamics import (CONTACT, FORCE, PROPRIO, EnhancedContactModeGNN,
                                      apply_enhanced_step, contact_modes)
from jenga_d6_set_response import sha256
from jenga_d7_explicit_curve import curve_targets
from jenga_d12_model_ladder import load_records, response_normalizers
from jenga_d13_monitor_evidence import evidence_rows, physical_evidence
from jenga_reobservation_interface import compare_nested
from jenga_w5_eval import load_model
from state_dynamics import rollout

PROTOCOL = ROOT / "results/jenga/d20_enhanced_hybrid_protocol.json"
DATA = ROOT / "results/jenga/d20_enhanced_trajectory_data.npz"
STAGE1 = ROOT / "results/jenga/d20_enhanced_hybrid_result.json"
OUTPUT = ROOT / "results/jenga/d20_enhanced_hybrid"
REPORT = ROOT / "results/jenga/d20_h8_geometry_result.json"
BASE_CONTINUOUS = tuple(range(45)) + (57, 58, 59)


def load_all(protocol, data_path):
    with np.load(data_path, allow_pickle=False) as stored:
        archive = {"start": stored["start"], "trajectory": stored["trajectory"]}
    manifest = json.loads((ROOT / protocol["inputs"]["manifest"]).read_text())
    directories = {"d3": ROOT / protocol["inputs"]["d3_directory"],
                   "d12": ROOT / protocol["inputs"]["d12_directory"]}
    loaded = load_records(manifest, directories, {"fit", "episode_validation",
                                                  "configuration_validation", "joint_validation"})
    actions = loaded[1].reshape(len(loaded[1]), 12, 38, 4)
    physical = loaded[2]
    folds = {name: np.asarray([i for i, row in enumerate(manifest["records"])
                              if row["fold"] == name])
             for name in ("fit", "episode_validation", "configuration_validation", "joint_validation")}
    return archive, actions, physical, folds, loaded


def model_scales(norms, device):
    return tuple(torch.as_tensor(norms[name], device=device) for name in
                 ("block_scale", "grip_scale", "proprio_scale", "force_mean", "force_scale"))


def model_inputs(norms, device):
    return tuple(torch.as_tensor(norms[name], device=device) for name in
                 ("state_mean", "state_scale", "action_mean", "action_scale"))


def unroll_enhanced(model, start, actions, input_norms, scales, hard_contacts=False):
    current = start; states = []; outputs = []
    for time_index in range(actions.shape[1]):
        output = model(current, actions[:, time_index], *input_norms)
        current = apply_enhanced_step(current, actions[:, time_index], output, scales,
                                      hard_contacts=hard_contacts)
        states.append(current); outputs.append(output)
    return torch.stack(states, 1), outputs


def response_geometry(predicted, truth, state_scale):
    groups = predicted.shape[0]; pose_scale = state_scale[:27]
    p = predicted[..., :27] / pose_scale; q = truth[..., :27] / pose_scale
    p = p.reshape(groups, 6, 2, predicted.shape[-2], 27)
    q = q.reshape(groups, 6, 2, truth.shape[-2], 27)
    p_gap = (p[:, :, 0] - p[:, :, 1]).norm(dim=-1)
    q_gap = (q[:, :, 0] - q[:, :, 1]).norm(dim=-1)
    return ((p_gap - q_gap) ** 2).mean()


def train_h8(model, archive, actions_np, fit_indices, norms_np, protocol, device):
    cfg = protocol["stages"]["h8"]; weights = cfg["loss_weights"]
    input_norms = model_inputs(norms_np, device); scales = model_scales(norms_np, device)
    state_scale = torch.as_tensor(norms_np["state_scale"], device=device)
    force_mean, force_scale = scales[-2:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng([20, 8]); history = []; began = time.time()
    for epoch in range(1, cfg["rollout_finetune_epochs"] + 1):
        order = rng.permutation(fit_indices); aggregates = {}; batches = 0; model.train()
        for offset in range(0, len(order), cfg["batch_groups"]):
            groups = order[offset:offset + cfg["batch_groups"]]; count = len(groups)
            start_np = np.repeat(archive["start"][groups, None], 12, axis=1).reshape(-1, 181)
            truth_np = archive["trajectory"][groups, :, :, :8].reshape(-1, 8, 181)
            action_np = actions_np[groups, :, :8].reshape(-1, 8, 4)
            start = torch.as_tensor(start_np, device=device); truth = torch.as_tensor(truth_np, device=device)
            control = torch.as_tensor(action_np, device=device)
            predicted, outputs = unroll_enhanced(model, start, control, input_norms, scales, False)
            base = ((predicted[..., BASE_CONTINUOUS] - truth[..., BASE_CONTINUOUS])
                    / state_scale[list(BASE_CONTINUOUS)]).square().mean()
            proprio = ((predicted[..., PROPRIO] - truth[..., PROPRIO])
                        / state_scale[PROPRIO]).square().mean()
            force = (((torch.log1p(predicted[..., FORCE]) - force_mean) / force_scale
                      - (torch.log1p(truth[..., FORCE]) - force_mean) / force_scale) ** 2).mean()
            mode_terms = []
            for t, output in enumerate(outputs):
                previous = start[:, CONTACT] if t == 0 else truth[:, t - 1, CONTACT]
                target = contact_modes(previous, truth[:, t, CONTACT])
                mode_terms.append(F.cross_entropy(output["contact_mode_logits"].reshape(-1, 4),
                                                   target.reshape(-1)))
            mode = torch.stack(mode_terms).mean()
            geometry = response_geometry(predicted.reshape(count, 12, 8, 181),
                                         truth.reshape(count, 12, 8, 181), state_scale)
            terms = {"base": weights["base_state_trajectory"] * base,
                     "proprio": weights["proprioception_trajectory"] * proprio,
                     "contact_mode": weights["supervised_contact_mode"] * mode,
                     "force": weights["force_and_impulse_trajectory"] * force,
                     "geometry": weights["pairwise_action_response_geometry"] * geometry}
            loss = sum(terms.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); batches += 1
            for name, value in terms.items(): aggregates[name] = aggregates.get(name, 0.) + float(value.detach())
        row = {"epoch": epoch, **{name: value / batches for name, value in aggregates.items()}}
        row["total"] = sum(value for name, value in row.items() if name != "epoch")
        history.append(row); print(f"D20 H8 {row}", flush=True)
    return history, time.time() - began


def correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0: return 0.
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_h8(model, d2, d2_scale, archive, actions_np, indices, norms_np, device, batch_groups=16):
    input_norms = model_inputs(norms_np, device); scales = model_scales(norms_np, device)
    state_scale = torch.as_tensor(norms_np["state_scale"], device=device); squared = [0., 0., 0]
    enhanced_gaps, d2_gaps, true_gaps = [], [], []
    model.eval(); d2.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_groups):
            groups = indices[offset:offset + batch_groups]; count = len(groups)
            start_np = np.repeat(archive["start"][groups, None], 12, axis=1).reshape(-1, 181)
            action_np = actions_np[groups, :, :8].reshape(-1, 8, 4)
            truth = torch.as_tensor(archive["trajectory"][groups, :, :, :8].reshape(-1, 8, 181), device=device)
            start = torch.as_tensor(start_np, device=device); control = torch.as_tensor(action_np, device=device)
            enhanced, _ = unroll_enhanced(model, start, control, input_norms, scales, True)
            d2_pred = rollout(d2, start[:, :61], control, d2_scale)
            target = truth[:, -1, :27]; e = enhanced[:, -1, :27]; d = d2_pred[:, -1, :27]
            squared[0] += float((((e - target) / state_scale[:27]) ** 2).sum())
            squared[1] += float((((d - target) / state_scale[:27]) ** 2).sum())
            squared[2] += target.numel()
            def gaps(value):
                value = (value / state_scale[:27]).reshape(count, 6, 2, 27)
                return (value[:, :, 0] - value[:, :, 1]).norm(dim=-1).cpu().numpy().reshape(-1)
            enhanced_gaps.append(gaps(e)); d2_gaps.append(gaps(d)); true_gaps.append(gaps(target))
    eg, dg, tg = map(np.concatenate, (enhanced_gaps, d2_gaps, true_gaps))
    e_rmse, d_rmse = np.sqrt(squared[0] / squared[2]), np.sqrt(squared[1] / squared[2])
    e_corr, d_corr = correlation(eg, tg), correlation(dg, tg)
    return {"groups": len(indices), "pose_nrmse": float(e_rmse), "d2_pose_nrmse": float(d_rmse),
            "pose_nrmse_ratio_to_d2": float(e_rmse / d_rmse),
            "pair_distance_correlation": e_corr, "d2_pair_distance_correlation": d_corr,
            "pair_distance_correlation_gain_over_d2": e_corr - d_corr}


def predicted_trajectories(model, d2, d2_scale, archive, actions_np, indices, norms_np, device,
                           batch_groups=8):
    input_norms = model_inputs(norms_np, device); scales = model_scales(norms_np, device)
    enhanced_rows, d2_rows = [], []; model.eval(); d2.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_groups):
            groups = indices[offset:offset + batch_groups]; count = len(groups)
            start_np = np.repeat(archive["start"][groups, None], 12, axis=1).reshape(-1, 181)
            action_np = actions_np[groups].reshape(-1, 38, 4)
            start = torch.as_tensor(start_np, device=device); control = torch.as_tensor(action_np, device=device)
            enhanced, _ = unroll_enhanced(model, start, control, input_norms, scales, True)
            d2_pred = rollout(d2, start[:, :61], control, d2_scale)
            enhanced_rows.append(enhanced[:, :, :61].cpu().numpy().reshape(count, 6, 2, 38, 61))
            d2_rows.append(d2_pred.cpu().numpy().reshape(count, 6, 2, 38, 61))
    return np.concatenate(enhanced_rows), np.concatenate(d2_rows)


def geometry_metrics(actions, physical, enhanced, d2, response_scale):
    target_curve = curve_targets(physical, response_scale); enhanced_curve = curve_targets(enhanced, response_scale)
    d2_curve = curve_targets(d2, response_scale)
    target_evidence = physical_evidence(actions, physical, response_scale)
    enhanced_evidence = physical_evidence(actions, enhanced, response_scale)
    d2_evidence = physical_evidence(actions, d2, response_scale)
    sign_e = np.sign(target_evidence) == np.sign(enhanced_evidence)
    sign_d = np.sign(target_evidence) == np.sign(d2_evidence)
    nested_e = compare_nested(evidence_rows(target_evidence), evidence_rows(enhanced_evidence))
    nested_d = compare_nested(evidence_rows(target_evidence), evidence_rows(d2_evidence))
    return {"curve_mse": float(np.mean((enhanced_curve - target_curve) ** 2)),
            "d2_curve_mse": float(np.mean((d2_curve - target_curve) ** 2)),
            "boundary_sign_agreement": float(sign_e[:, :2].mean()),
            "d2_boundary_sign_agreement": float(sign_d[:, :2].mean()),
            "commitment_sign_agreement": float(sign_e[:, 2].mean()),
            "d2_commitment_sign_agreement": float(sign_d[:, 2].mean()),
            "final_agreement": nested_e["alarm"]["agreement"],
            "d2_final_agreement": nested_d["alarm"]["agreement"],
            "nested": nested_e, "d2_nested": nested_d}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--stage1", default=str(STAGE1))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", default="auto"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    stage1 = json.loads(Path(args.stage1).read_text())
    if stage1["protocol_sha256"] != sha256(protocol_path) or not stage1["stage1"]["passes"]:
        raise SystemExit("D20 H8 is not authorized by the matching Stage-1 result")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
             "cpu" if args.device == "auto" else args.device)
    archive, actions, physical, folds, loaded = load_all(protocol, args.data)
    checkpoint = torch.load(ROOT / stage1["checkpoint"], map_location="cpu", weights_only=False)
    if sha256(ROOT / stage1["checkpoint"]) != stage1["checkpoint_sha256"]:
        raise SystemExit("D20 Stage-1 checkpoint differs")
    model = EnhancedContactModeGNN(protocol["model"]["hidden"], protocol["model"]["graph_rounds"])
    model.load_state_dict(checkpoint["model"]); model.to(device); norms_np = checkpoint["normalizers"]
    history, seconds = train_h8(model, archive, actions, folds["fit"], norms_np, protocol, device)
    d2, d2_scale = load_model(ROOT / protocol["inputs"]["d2_checkpoint"], device)
    h8 = {fold: evaluate_h8(model, d2, d2_scale, archive, actions, folds[fold], norms_np, device)
          for fold in ("episode_validation", "configuration_validation")}
    cfg = protocol["stages"]["h8"]
    h8_gates = {fold: {"checks": {
        "pose": value["pose_nrmse_ratio_to_d2"] <= cfg["maximum_h8_pose_nrmse_ratio_to_d2"],
        "pair_geometry": value["pair_distance_correlation_gain_over_d2"] >= cfg["minimum_pair_distance_correlation_gain_over_d2"]}}
        for fold, value in h8.items()}
    for value in h8_gates.values(): value["passes"] = all(value["checks"].values())
    h8_passes = all(value["passes"] for value in h8_gates.values())
    geometry = None; geometry_passes = False; monitor_passes = False
    if h8_passes:
        response_scale = response_normalizers(*loaded[:3])[4]
        geometry = {}
        for fold in ("episode_validation", "configuration_validation"):
            indices = folds[fold]; enhanced, d2_pred = predicted_trajectories(
                model, d2, d2_scale, archive, actions, indices, norms_np, device)
            metrics = geometry_metrics(loaded[1][indices], physical[indices], enhanced, d2_pred,
                                       response_scale)
            checks = {"curve_mse": metrics["curve_mse"] < metrics["d2_curve_mse"],
                      "boundary_sign": metrics["boundary_sign_agreement"] > metrics["d2_boundary_sign_agreement"],
                      "commitment_sign": metrics["commitment_sign_agreement"] > metrics["d2_commitment_sign_agreement"]}
            geometry[fold] = {"metrics": metrics, "checks": checks, "passes": all(checks.values())}
        geometry_passes = all(value["passes"] for value in geometry.values())
        if geometry_passes:
            monitor_passes = all(value["metrics"]["final_agreement"] >
                                 value["metrics"]["d2_final_agreement"] for value in geometry.values())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); saved = output / "h8_seed1.pt"
    torch.save({"model": model.state_dict(), "model_config": protocol["model"],
                "normalizers": norms_np, "protocol_sha256": sha256(protocol_path)}, saved)
    decision = ("stop_at_h8" if not h8_passes else "stop_at_geometry" if not geometry_passes
                else "stop_at_monitor" if not monitor_passes else "authorize_prospective_dev")
    result = {"protocol_sha256": sha256(protocol_path), "created": datetime.now(timezone.utc).isoformat(),
              "device": device, "stage2": {"training_seconds": seconds, "history": history,
              "evaluations": h8, "gates": h8_gates, "passes": h8_passes},
              "stage3": geometry, "stage3_passes": geometry_passes,
              "stage4_passes": monitor_passes, "decision": decision,
              "checkpoint": str(saved.relative_to(ROOT)), "checkpoint_sha256": sha256(saved),
              "dev_or_test_read": False, "task_labels_used": False}
    report.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
