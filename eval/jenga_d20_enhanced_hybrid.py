"""Train and gate D20 enhanced-state supervised-contact dynamics stage by stage."""
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
                                      contact_modes, continuous_deltas)
from jenga_d6_set_response import sha256
from jenga_d12_model_ladder import load_records
from jenga_w5_eval import load_model
from state_dynamics import apply_step, contact_targets

PROTOCOL = ROOT / "results/jenga/d20_enhanced_hybrid_protocol.json"
DATA = ROOT / "results/jenga/d20_enhanced_trajectory_data.npz"
OUTPUT = ROOT / "results/jenga/d20_enhanced_hybrid"
REPORT = ROOT / "results/jenga/d20_enhanced_hybrid_result.json"


def gather(data, actions, group_indices, flat_indices):
    per_group = 12 * 38
    group_local = flat_indices // per_group; remainder = flat_indices % per_group
    path = remainder // 38; time_index = remainder % 38
    group = group_indices[group_local]
    following = data["trajectory"][group, path // 2, path % 2, time_index]
    current = np.empty_like(following)
    first = time_index == 0
    current[first] = data["start"][group[first]]
    if np.any(~first):
        prior_time = time_index[~first] - 1
        current[~first] = data["trajectory"][group[~first], path[~first] // 2,
                                                    path[~first] % 2, prior_time]
    action = actions[group, path, time_index]
    return current, action, following


def normalizers(data, actions, fit_indices, sample_batch=4096):
    # Streaming statistics avoid materializing two copies of the 181-D trajectory corpus.
    sums = {name: None for name in ("state", "state2", "action", "action2", "block2", "grip2",
                                    "proprio2", "force", "force2")}
    count = 0; mode_counts = np.zeros(4, np.int64)
    total = len(fit_indices) * 12 * 38
    for start in range(0, total, sample_batch):
        flat = np.arange(start, min(start + sample_batch, total))
        current, action, following = gather(data, actions, fit_indices, flat)
        current_t = torch.from_numpy(current); following_t = torch.from_numpy(following)
        block, grip, proprio = continuous_deltas(current_t, following_t)
        values = {"state": current, "state2": current ** 2, "action": action,
                  "action2": action ** 2, "block2": block.numpy().reshape(-1, 15) ** 2,
                  "grip2": grip.numpy() ** 2, "proprio2": proprio.numpy() ** 2,
                  "force": np.log1p(following[:, FORCE]),
                  "force2": np.log1p(following[:, FORCE]) ** 2}
        for name, value in values.items():
            aggregate = value.sum(0, dtype=np.float64)
            sums[name] = aggregate if sums[name] is None else sums[name] + aggregate
        labels = contact_modes(current_t[:, CONTACT], following_t[:, CONTACT]).numpy()
        mode_counts += np.bincount(labels.reshape(-1), minlength=4)
        count += len(flat)
    def mean_std(name, square):
        mean = sums[name] / count; variance = np.maximum(sums[square] / count - mean ** 2, 0.)
        return mean.astype(np.float32), np.sqrt(variance).clip(1e-7).astype(np.float32)
    state_mean, state_scale = mean_std("state", "state2")
    action_mean, action_scale = mean_std("action", "action2")
    force_mean, force_scale = mean_std("force", "force2")
    # Delta targets are zero-mean; RMS is the correct scale used by both arms.
    block_scale = np.sqrt(sums["block2"] / (count * 3)).clip(1e-7).astype(np.float32)
    grip_scale = np.sqrt(sums["grip2"] / count).clip(1e-7).astype(np.float32)
    proprio_scale = np.sqrt(sums["proprio2"] / count).clip(1e-7).astype(np.float32)
    class_weight = np.sqrt(mode_counts.sum() / np.maximum(mode_counts, 1))
    class_weight = (class_weight / class_weight.mean()).astype(np.float32)
    return {"state_mean": state_mean, "state_scale": state_scale,
            "action_mean": action_mean, "action_scale": action_scale,
            "block_scale": block_scale, "grip_scale": grip_scale,
            "proprio_scale": proprio_scale, "force_mean": force_mean,
            "force_scale": force_scale, "contact_class_weight": class_weight,
            "contact_mode_counts": mode_counts}


def tensor_norms(norms, device):
    return {name: torch.as_tensor(value, device=device) for name, value in norms.items()
            if name != "contact_mode_counts"}


def losses(model, current, action, following, norms, weights):
    output = model(current, action, norms["state_mean"], norms["state_scale"],
                   norms["action_mean"], norms["action_scale"])
    block, grip, proprio = continuous_deltas(current, following)
    continuous = ((output["block_mean"] - block / norms["block_scale"]) ** 2).mean()
    continuous = continuous + ((output["gripper_mean"] - grip / norms["grip_scale"]) ** 2).mean()
    proprio_loss = ((output["proprio_delta"] - proprio / norms["proprio_scale"]) ** 2).mean()
    modes = contact_modes(current[:, CONTACT], following[:, CONTACT])
    mode_loss = F.cross_entropy(output["contact_mode_logits"].reshape(-1, 4), modes.reshape(-1),
                                weight=norms["contact_class_weight"])
    force_target = (torch.log1p(following[:, FORCE]) - norms["force_mean"]) / norms["force_scale"]
    force_loss = ((output["force_next"] - force_target) ** 2).mean()
    return output, {"continuous": weights["continuous_base_delta"] * continuous,
                    "proprio": weights["proprioception_delta"] * proprio_loss,
                    "contact_mode": weights["supervised_contact_mode"] * mode_loss,
                    "force": weights["force_and_impulse_next_state"] * force_loss}


def event_f1(target, predicted):
    values = []
    for label in (2, 3):
        tp = np.sum((target == label) & (predicted == label))
        fp = np.sum((target != label) & (predicted == label))
        fn = np.sum((target == label) & (predicted != label))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values)), {"created": float(values[0]), "lost": float(values[1])}


def evaluate_one_step(model, d2, d2_scale, data, actions, indices, norms, device, batch=4096):
    totals = {"enhanced_squared": 0., "d2_squared": 0., "count": 0}
    target_modes, enhanced_modes, d2_modes = [], [], []
    model.eval(); total = len(indices) * 12 * 38
    with torch.no_grad():
        for start in range(0, total, batch):
            flat = np.arange(start, min(start + batch, total))
            current_np, action_np, following_np = gather(data, actions, indices, flat)
            current = torch.as_tensor(current_np, device=device)
            action = torch.as_tensor(action_np, device=device)
            following = torch.as_tensor(following_np, device=device)
            output = model(current, action, norms["state_mean"], norms["state_scale"],
                           norms["action_mean"], norms["action_scale"])
            block, grip, _ = continuous_deltas(current, following)
            enhanced_error = torch.cat(((output["block_mean"] - block / norms["block_scale"]).reshape(len(flat), -1),
                                        output["gripper_mean"] - grip / norms["grip_scale"]), 1)
            d2_output = d2(current[:, :61], action)
            d2_error = torch.cat((((d2_output["block_mean"] * d2_scale[0] - block)
                                   / norms["block_scale"]).reshape(len(flat), -1),
                                  (d2_output["gripper_mean"] * d2_scale[1] - grip)
                                  / norms["grip_scale"]), 1)
            totals["enhanced_squared"] += float((enhanced_error ** 2).sum())
            totals["d2_squared"] += float((d2_error ** 2).sum()); totals["count"] += enhanced_error.numel()
            target = contact_modes(current[:, CONTACT], following[:, CONTACT])
            enhanced_prediction = output["contact_mode_logits"].argmax(-1)
            d2_next = apply_step(current[:, :61], action, d2_output, d2_scale)
            d2_prediction = contact_modes(current[:, CONTACT], d2_next[:, CONTACT])
            target_modes.append(target.cpu().numpy()); enhanced_modes.append(enhanced_prediction.cpu().numpy())
            d2_modes.append(d2_prediction.cpu().numpy())
    target = np.concatenate(target_modes); enhanced_pred = np.concatenate(enhanced_modes)
    d2_pred = np.concatenate(d2_modes); enhanced_f1, enhanced_detail = event_f1(target, enhanced_pred)
    d2_f1, d2_detail = event_f1(target, d2_pred)
    enhanced_rmse = np.sqrt(totals["enhanced_squared"] / totals["count"])
    d2_rmse = np.sqrt(totals["d2_squared"] / totals["count"])
    return {"transitions": total, "continuous_nrmse": float(enhanced_rmse),
            "d2_continuous_nrmse": float(d2_rmse), "continuous_nrmse_ratio_to_d2": float(enhanced_rmse / d2_rmse),
            "contact_event_macro_f1": enhanced_f1, "d2_contact_event_macro_f1": d2_f1,
            "contact_event_macro_f1_gain_over_d2": enhanced_f1 - d2_f1,
            "contact_event_f1": enhanced_detail, "d2_contact_event_f1": d2_detail}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    for key, hash_key in (("manifest", "manifest_sha256"), ("d17_start_state", "d17_start_state_sha256"),
                          ("d2_checkpoint", "d2_checkpoint_sha256"), ("monitor_targets", "monitor_targets_sha256")):
        if sha256(ROOT / protocol["inputs"][key]) != protocol["inputs"][hash_key]:
            raise SystemExit(f"frozen input differs: {key}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
             "cpu" if args.device == "auto" else args.device)
    integrity_path = Path(args.data).with_name("d20_enhanced_trajectory_integrity.json")
    integrity = json.loads(integrity_path.read_text())
    if not integrity["passes"] or integrity["protocol_sha256"] != sha256(protocol_path):
        raise SystemExit("D20 enhanced trajectory integrity is absent or belongs to another protocol")
    if integrity["output_sha256"] != sha256(args.data):
        raise SystemExit("D20 enhanced trajectory data hash differs from its integrity report")
    with np.load(args.data, allow_pickle=False) as stored:
        archive = {"start": stored["start"], "trajectory": stored["trajectory"]}
    manifest = json.loads((ROOT / protocol["inputs"]["manifest"]).read_text())
    directories = {"d3": ROOT / protocol["inputs"]["d3_directory"],
                   "d12": ROOT / protocol["inputs"]["d12_directory"]}
    all_data = load_records(manifest, directories, {"fit", "episode_validation",
                                                    "configuration_validation", "joint_validation"})
    actions = all_data[1].reshape(len(all_data[1]), 12, 38, 4)
    if len(actions) != len(archive["start"]): raise SystemExit("D20 data/manifest ordering differs")
    folds = {name: np.asarray([i for i, row in enumerate(manifest["records"]) if row["fold"] == name])
             for name in ("fit", "episode_validation", "configuration_validation", "joint_validation")}
    norms_np = normalizers(archive, actions, folds["fit"]); norms = tensor_norms(norms_np, device)
    torch.manual_seed(protocol["model"]["seed"]); model = EnhancedContactModeGNN(
        protocol["model"]["hidden"], protocol["model"]["graph_rounds"]).to(device)
    stage_cfg = protocol["stages"]["one_step"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg["learning_rate"])
    rng = np.random.default_rng([20, 1])
    total = len(folds["fit"]) * 12 * 38; batch_size = stage_cfg["batch_transitions"]
    history = []; began = time.time()
    for epoch in range(1, stage_cfg["train_epochs"] + 1):
        order = rng.permutation(total); aggregates = {}
        model.train()
        for start in range(0, total, batch_size):
            current_np, action_np, following_np = gather(
                archive, actions, folds["fit"], order[start:start + batch_size])
            current = torch.as_tensor(current_np, device=device); action = torch.as_tensor(action_np, device=device)
            following = torch.as_tensor(following_np, device=device)
            _, terms = losses(model, current, action, following, norms, stage_cfg["loss_weights"])
            loss = sum(terms.values())
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            for name, value in terms.items(): aggregates[name] = aggregates.get(name, 0.) + float(value.detach())
        row = {"epoch": epoch, **{name: value / np.ceil(total / batch_size) for name, value in aggregates.items()}}
        row["total"] = sum(value for name, value in row.items() if name != "epoch")
        history.append(row); print(f"D20 one-step {row}", flush=True)
    d2, d2_scale = load_model(ROOT / protocol["inputs"]["d2_checkpoint"], device)
    evaluations = {fold: evaluate_one_step(model, d2, d2_scale, archive, actions, folds[fold], norms, device)
                   for fold in ("episode_validation", "configuration_validation")}
    gate_cfg = protocol["stages"]["one_step"]["gate_on_both_primary_axes"]
    gates = {fold: {"checks": {
        "continuous": row["continuous_nrmse_ratio_to_d2"] <= gate_cfg["maximum_continuous_nrmse_ratio_to_d2"],
        "contact_events": row["contact_event_macro_f1_gain_over_d2"] >= gate_cfg["minimum_contact_event_macro_f1_gain_over_d2"]}}
        for fold, row in evaluations.items()}
    for value in gates.values(): value["passes"] = all(value["checks"].values())
    passed = all(value["passes"] for value in gates.values()); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "one_step_seed1.pt"
    torch.save({"model": model.state_dict(), "model_config": protocol["model"],
                "normalizers": norms_np, "protocol_sha256": sha256(protocol_path)}, checkpoint)
    result = {"protocol": protocol, "protocol_sha256": sha256(protocol_path),
              "created": datetime.now(timezone.utc).isoformat(), "device": device,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trajectory_data_sha256": integrity["output_sha256"],
              "stage1": {"training_seconds": time.time() - began, "history": history,
                         "evaluations": evaluations, "gates": gates, "passes": passed},
              "later_stages_executed": False, "decision": ("authorize_h8" if passed else "stop_at_one_step"),
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
              "dev_or_test_read": False, "task_labels_used": False}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"evaluations": evaluations, "gates": gates, "decision": result["decision"]}, indent=2))


if __name__ == "__main__": main()
