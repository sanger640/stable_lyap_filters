"""Frozen D14 Stage-1 anonymous object/contact graph experiment."""
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

from jenga_d6_set_response import sha256, tensors
from jenga_d7_explicit_curve import curve_targets
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_d12_model_ladder import load_records, response_normalizers
from jenga_d13_monitor_evidence import (decode_evidence, evaluate_evidence, gate,
                                        predict_curve)
from set_response_model import ObjectGraphJointMonitorEvidenceModel


PROTOCOL = ROOT / "results/jenga/d14_object_graph_protocol.json"
REPORT = ROOT / "results/jenga/d14_object_graph_result.json"
OUTPUT = ROOT / "results/jenga/d14_object_graph"


def permute_objects(state, order):
    """Relabel all object pose, velocity, and contact channels consistently."""
    state = state.clone(); source = state.clone(); order = list(order)
    for base, width in ((0, 3), (9, 6), (27, 6)):
        values = source[:, base:base + 3 * width].reshape(len(state), 3, width)
        state[:, base:base + 3 * width] = values[:, order].reshape(len(state), -1)
    old_contacts = source[:, 45:57]; new_contacts = state[:, 45:57]
    pair_index = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        new_contacts[:, pair_index[(first, second)]] = old_contacts[
            :, pair_index[tuple(sorted((order[first], order[second])))]].clone()
    for offset in (3, 6, 9):
        new_contacts[:, offset:offset + 3] = old_contacts[:, offset:offset + 3][:, order]
    return state


def output_max_abs(first, second):
    return max(float(torch.max(torch.abs(left - right)))
               for left, right in zip(first, second))


def invariance(model, data, norms, device):
    count = min(2, len(data[0]))
    state = torch.as_tensor(data[0][:count], device=device)
    actions = torch.as_tensor(data[1][:count], device=device)
    level_order = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    object_order = (2, 0, 1)
    translated_state = state.clone(); translated_actions = actions.clone()
    offset = torch.tensor([.071, -.043], device=device)
    positions = translated_state[:, :9].reshape(count, 3, 3)
    positions[..., :2] += offset
    translated_state[:, 57:59] += offset
    translated_actions[..., :2] += offset
    model.eval()
    with torch.no_grad():
        original = model(state, actions, *norms[:4])
        endpoint = model(state, actions.flip(2), *norms[:4])
        levels = model(state, actions[:, level_order], *norms[:4])
        objects = model(permute_objects(state, object_order), actions, *norms[:4])
        translated = model(translated_state, translated_actions, *norms[:4])
    # Level-valued curve outputs must be unpermuted before comparison; group outputs are invariant.
    inverse = torch.argsort(level_order)
    level_aligned = tuple(value[:, inverse] if value.ndim >= 3 and value.shape[1] == 6 else value
                          for value in levels)
    return {"endpoint_swap_max_abs": output_max_abs(original, endpoint),
            "level_permutation_max_abs": output_max_abs(original, level_aligned),
            "object_permutation_max_abs": output_max_abs(original, objects),
            "horizontal_translation_max_abs": output_max_abs(original, translated)}


def predict_evidence(model, data, norms, target_mean, target_scale, device):
    model.eval()
    with torch.no_grad():
        normalized = model(torch.as_tensor(data[0], device=device),
                           torch.as_tensor(data[1], device=device), *norms[:4])[-1]
    return decode_evidence(normalized.cpu().numpy(), target_mean, target_scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {report}")
    protocol_path = Path(args.protocol); protocol = json.loads(protocol_path.read_text())
    protocol_hash = sha256(protocol_path)
    data_cfg = protocol["data"]; controls_cfg = protocol["frozen_controls"]
    for relative, expected in (
            (data_cfg["manifest"], data_cfg["manifest_sha256"]),
            (data_cfg["targets"], data_cfg["targets_sha256"]),
            (controls_cfg["d13_result"], controls_cfg["d13_result_sha256"]),
            (controls_cfg["ordered_vector_joint_checkpoint"],
             controls_cfg["ordered_vector_joint_checkpoint_sha256"])):
        if sha256(ROOT / relative) != expected:
            raise SystemExit(f"frozen input hash differs: {relative}")
    manifest = json.loads((ROOT / data_cfg["manifest"]).read_text())
    directories = {"d3": ROOT / data_cfg["d3_directory"],
                   "d12": ROOT / data_cfg["d12_directory"]}
    fold_names = ("fit", "episode_validation", "configuration_validation", "joint_validation")
    folds = {fold: load_records(manifest, directories, {fold}) for fold in fold_names}
    if any(len(folds[fold][0]) != data_cfg["groups"][fold] for fold in fold_names):
        raise SystemExit("fold counts differ from frozen protocol")
    normalizer_np = response_normalizers(*folds["fit"][:3])
    norms = tensors(normalizer_np, "cuda" if (args.device == "auto" and torch.cuda.is_available())
                    else ("cpu" if args.device == "auto" else args.device))
    device = norms[0].device
    target_file = np.load(ROOT / data_cfg["targets"], allow_pickle=False)
    target_mean = target_file["target_mean"]; target_scale = target_file["target_scale"]
    raw_targets = {fold: target_file[f"{fold}_raw"] for fold in fold_names}; target_file.close()
    normalized_target = ((np.sign(raw_targets["fit"]) * np.log1p(np.abs(raw_targets["fit"]))
                          - target_mean) / target_scale).astype(np.float32)
    response_scale = normalizer_np[4]
    curve_target = curve_targets(folds["fit"][2], response_scale)
    target_curve = torch.as_tensor(curve_target, device=device)
    target_amplitude, target_shape, nonzero = factor_targets(target_curve)
    evidence_target = torch.as_tensor(normalized_target, device=device)
    state = torch.as_tensor(folds["fit"][0], device=device)
    actions = torch.as_tensor(folds["fit"][1], device=device)

    config = protocol["model"]; optimization = protocol["optimization"]
    torch.manual_seed(optimization["pilot_seed"])
    model = ObjectGraphJointMonitorEvidenceModel(
        hidden=config["hidden"], heads=config["heads"], layers=config["layers"],
        graph_rounds=config["graph_rounds"], action_horizon=config["action_horizon"],
        curve_steps=config["curve_steps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    rng = np.random.default_rng([optimization["pilot_seed"], 14]); history = []; began = time.time()
    for step in range(1, optimization["steps"] + 1):
        chosen = torch.as_tensor(rng.choice(len(state), optimization["group_batch_size"],
                                            replace=False), device=device)
        model.train(); predicted, amplitude, shape, evidence = model(
            state[chosen], actions[chosen], *norms[:4])
        losses = factor_losses(amplitude, shape, target_amplitude[chosen],
                               target_shape[chosen], nonzero[chosen])
        losses["evidence"] = (evidence - evidence_target[chosen]).square().mean()
        loss = sum(losses.values()); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 1000 == 0 or step == optimization["steps"]:
            row = {"step": step, "total": float(loss.detach()),
                   **{name: float(value.detach()) for name, value in losses.items()}}
            history.append(row); print(f"D14 graph: {row}", flush=True)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "graph_joint_seed1.pt"
    torch.save({"model": model.state_dict(), "seed": optimization["pilot_seed"],
                "protocol_sha256": protocol_hash}, checkpoint)
    evaluations = {}
    for fold, values in folds.items():
        predicted_raw = predict_evidence(
            model, values, norms, target_mean, target_scale, device)
        evaluations[fold] = evaluate_evidence(
            raw_targets[fold], predicted_raw, target_mean, target_scale)
    invariants = invariance(model, folds["fit"], norms, device)
    d13 = json.loads((ROOT / controls_cfg["d13_result"]).read_text())
    curve_control = d13["curve_baseline"]["evaluations"]
    vector_control = d13["arms"]["joint"]["evaluations"]
    gate_invariants = {"max_abs": max(invariants.values())}
    gates = {fold: gate(curve_control[fold], evaluations[fold],
                        protocol["gate_on_each_primary_axis"], gate_invariants)
             for fold in protocol["primary_validation_axes"]}
    passes = all(value["passes"] for value in gates.values())
    vector_deltas = {fold: {
        "normalized_evidence_mse": evaluations[fold]["normalized_evidence_mse"]
                                   - vector_control[fold]["normalized_evidence_mse"],
        "mean_sign_agreement": evaluations[fold]["mean_channel_sign_agreement"]
                               - vector_control[fold]["mean_channel_sign_agreement"],
        "boundary_recall": evaluations[fold]["nested"]["boundary"]["recall"]
                           - vector_control[fold]["nested"]["boundary"]["recall"],
        "boundary_added_positive_rate":
            evaluations[fold]["nested"]["boundary"]["added_positive_rate"]
            - vector_control[fold]["nested"]["boundary"]["added_positive_rate"],
        "final_recall": evaluations[fold]["nested"]["alarm"]["recall"]
                        - vector_control[fold]["nested"]["alarm"]["recall"],
        "final_added_positive_rate":
            evaluations[fold]["nested"]["alarm"]["added_positive_rate"]
            - vector_control[fold]["nested"]["alarm"]["added_positive_rate"]}
        for fold in protocol["primary_validation_axes"]}
    result = {
        "protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": protocol_hash, "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {"seconds": time.time() - began, "history": history,
                     "checkpoint": str(checkpoint.relative_to(ROOT)),
                     "checkpoint_sha256": sha256(checkpoint)},
        "invariance": invariants, "evaluations": evaluations,
        "ordered_vector_deltas": vector_deltas, "gates": gates,
        "stage1_passes": passes, "replication_authorized": passes,
        "stage2_push_collection_authorized": passes, "dev_or_test_read": False,
        "interpretation": ("graph_pass_authorize_replication" if passes else
                           "graph_failed_stop_before_push_collection")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"evaluations": evaluations, "ordered_vector_deltas": vector_deltas,
                      "invariance": invariants, "gates": gates, "stage1_passes": passes,
                      "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
