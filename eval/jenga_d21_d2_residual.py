"""Frozen D21 Stage 1: preserve D2 and learn one enhanced-state contact residual."""
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

from d2_enhanced_residual import D2EnhancedResidual
from jenga_d6_set_response import sha256
from jenga_d12_model_ladder import load_records
from jenga_d20_enhanced_hybrid import (evaluate_one_step, gather, losses, tensor_norms)
from jenga_w5_eval import load_model

PROTOCOL = ROOT / "results/jenga/d21_d2_enhanced_residual_protocol.json"
REPORT = ROOT / "results/jenga/d21_d2_enhanced_residual_result.json"
OUTPUT = ROOT / "results/jenga/d21_d2_enhanced_residual"


def identity_check(model, d2, d2_scale, archive, actions, fit_indices, norms, device):
    current_np, action_np, _ = gather(archive, actions, fit_indices, np.arange(4096))
    current = torch.as_tensor(current_np, device=device); action = torch.as_tensor(action_np, device=device)
    model.eval(); d2.eval()
    with torch.no_grad():
        residual = model(current, action, norms["state_mean"], norms["state_scale"],
                         norms["action_mean"], norms["action_scale"])
    block_error = torch.max(torch.abs(residual["block_physical"]
                                      - residual["base_block_physical"]))
    grip_error = torch.max(torch.abs(residual["gripper_physical"]
                                     - residual["base_gripper_physical"]))
    contact_exact = torch.equal(torch.isin(residual["contact_mode_logits"].argmax(-1),
                                           torch.tensor([1, 2], device=device)),
                                residual["base_contact_active_logits"] > 0)
    return {"block_physical_max_abs": float(block_error), "gripper_physical_max_abs": float(grip_error),
            "contact_decisions_exact": bool(contact_exact)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", default="auto"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text()); inputs = protocol["inputs"]
    for key, hash_key in (("manifest", "manifest_sha256"),
                          ("enhanced_trajectory_data", "enhanced_trajectory_data_sha256"),
                          ("enhanced_trajectory_integrity", "enhanced_trajectory_integrity_sha256"),
                          ("d2_checkpoint", "d2_checkpoint_sha256"),
                          ("d20_result", "d20_result_sha256"),
                          ("d20_checkpoint", "d20_checkpoint_sha256")):
        if sha256(ROOT / inputs[key]) != inputs[hash_key]: raise SystemExit(f"frozen input differs: {key}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
             "cpu" if args.device == "auto" else args.device)
    with np.load(ROOT / inputs["enhanced_trajectory_data"], allow_pickle=False) as stored:
        archive = {"start": stored["start"], "trajectory": stored["trajectory"]}
    manifest = json.loads((ROOT / inputs["manifest"]).read_text()); directories = {
        "d3": ROOT / inputs["d3_directory"], "d12": ROOT / inputs["d12_directory"]}
    loaded = load_records(manifest, directories, {"fit", "episode_validation",
                                                  "configuration_validation", "joint_validation"})
    actions = loaded[1].reshape(len(loaded[1]), 12, 38, 4)
    folds = {name: np.asarray([i for i, row in enumerate(manifest["records"]) if row["fold"] == name])
             for name in ("fit", "episode_validation", "configuration_validation")}
    d20_checkpoint = torch.load(ROOT / inputs["d20_checkpoint"], map_location="cpu", weights_only=False)
    norms_np = d20_checkpoint["normalizers"]; norms = tensor_norms(norms_np, device)
    d2_state = torch.load(ROOT / inputs["d2_checkpoint"], map_location="cpu", weights_only=False)
    d2, d2_scale = load_model(ROOT / inputs["d2_checkpoint"], device)
    torch.manual_seed(protocol["model"]["seed"])
    model = D2EnhancedResidual(d2_state, d2_scale[0].cpu(), d2_scale[1].cpu(),
                               norms_np["block_scale"], norms_np["grip_scale"]).to(device)
    identity = identity_check(model, d2, d2_scale, archive, actions, folds["fit"], norms, device)
    tolerance = protocol["model"]["identity_tolerance"]
    identity_passes = (identity["block_physical_max_abs"] <= tolerance and
                       identity["gripper_physical_max_abs"] <= tolerance and
                       identity["contact_decisions_exact"])
    if not identity_passes: raise SystemExit(f"D21 identity initialization failed: {identity}")
    cfg = protocol["stage1"]; weights = cfg["loss_weights"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg["learning_rate"])
    rng = np.random.default_rng([21, 1]); total = len(folds["fit"]) * 12 * 38
    history = []; began = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        order = rng.permutation(total); aggregate = {}; batches = 0; model.train()
        for offset in range(0, total, cfg["batch_transitions"]):
            current_np, action_np, following_np = gather(
                archive, actions, folds["fit"], order[offset:offset + cfg["batch_transitions"]])
            current = torch.as_tensor(current_np, device=device); action = torch.as_tensor(action_np, device=device)
            following = torch.as_tensor(following_np, device=device)
            output, terms = losses(model, current, action, following, norms, weights)
            terms["residual_l2"] = weights["continuous_residual_l2"] * output[
                "continuous_residual"].square().mean()
            loss = sum(terms.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            optimizer.step(); batches += 1
            for name, value in terms.items(): aggregate[name] = aggregate.get(name, 0.) + float(value.detach())
        row = {"epoch": epoch, **{name: value / batches for name, value in aggregate.items()}}
        row["total"] = sum(value for name, value in row.items() if name != "epoch")
        history.append(row); print(f"D21 residual {row}", flush=True)
    evaluations = {fold: evaluate_one_step(model, d2, d2_scale, archive, actions, folds[fold], norms, device)
                   for fold in ("episode_validation", "configuration_validation")}
    d20 = json.loads((ROOT / inputs["d20_result"]).read_text())["stage1"]["evaluations"]
    gates = {}
    for fold, value in evaluations.items():
        checks = {"continuous_no_regression": value["continuous_nrmse_ratio_to_d2"] <=
                                               cfg["gate_on_each_primary_axis"]["maximum_continuous_nrmse_ratio_to_d2"],
                  "retain_d20_contact_f1": value["contact_event_macro_f1"] >=
                                            d20[fold]["contact_event_macro_f1"]}
        gates[fold] = {"checks": checks, "passes": all(checks.values()),
                       "d20_contact_f1_floor": d20[fold]["contact_event_macro_f1"]}
    passes = all(value["passes"] for value in gates.values()); output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_path = output_dir / "seed1.pt"
    torch.save({"model": model.state_dict(), "normalizers": norms_np,
                "protocol_sha256": sha256(protocol_path)}, checkpoint_path)
    result = {"protocol": protocol, "protocol_sha256": sha256(protocol_path),
              "created": datetime.now(timezone.utc).isoformat(), "device": device,
              "identity": identity, "identity_passes": identity_passes,
              "stage1": {"training_seconds": time.time() - began, "history": history,
                         "evaluations": evaluations, "gates": gates, "passes": passes},
              "decision": "authorize_h8" if passes else "stop_model_tweaking_line",
              "checkpoint": str(checkpoint_path.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint_path), "dev_or_test_read": False,
              "task_labels_used": False}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"identity": identity, "evaluations": evaluations, "gates": gates,
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__": main()
