"""Train and evaluate the frozen D7 explicit nested pair-curve head."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import boundary_refinement_alarm
from consequence_monitor import (commitment_delta_bic, persistence_delta_bic,
                                 whole_trajectory_separation_curves)
from jenga_d6_memorization import selected_nested, selected_rows
from jenga_d6_set_response import DATA, NESTED, evaluate, normalizers, sha256, tensors
from jenga_reobservation_interface import (compare_nested, nested_stage_rows, validation_rows)
from set_response_model import DirectSetResponseModel, NestedPairCurveHead


PROTOCOL = ROOT / "results/jenga/d7_explicit_curve_protocol.json"
REPORT = ROOT / "results/jenga/d7_explicit_curve_result.json"
CHECKPOINT = ROOT / "results/jenga/d7_explicit_curve/model.pt"


def curve_targets(traces, response_scale):
    traces = np.asarray(traces, np.float32)
    scale = np.asarray(response_scale, np.float32)[:45]
    difference = traces[..., 0, :, :45] / scale - traces[..., 1, :, :45] / scale
    early = np.sqrt(np.mean(difference[..., :18, :] ** 2, axis=(-2, -1)))
    full = np.sqrt(np.mean(difference ** 2, axis=(-2, -1)))
    curves = np.asarray([[
        whole_trajectory_separation_curves(group[level:level + 1])[0]
        for level in range(len(group))] for group in traces], np.float32)
    return np.log1p(np.concatenate([early[..., None], full[..., None], curves], axis=-1))


def loss_terms(predicted, target):
    return {
        "early_gap": (predicted[..., 0] - target[..., 0]).square().mean(),
        "full_gap": (predicted[..., 1] - target[..., 1]).square().mean(),
        "curve_action": (predicted[..., 2:11] - target[..., 2:11]).square().mean(),
        "curve_early_hold": (predicted[..., 11:21] - target[..., 11:21]).square().mean(),
        "curve_late_hold": (predicted[..., 21:41] - target[..., 21:41]).square().mean(),
    }


def decoded(predicted):
    return torch.expm1(predicted.clamp_min(0.))


def consequence_from_curves(curves, action_differences, early_boundary_bic, full_boundary_bic,
                            horizon=8, immediate_hold=5, late_hold=10):
    """Apply frozen Monitor v0 mathematics without modifying its frozen source file."""
    curves = np.asarray(curves, float); actions = np.asarray(action_differences, float)
    early = np.asarray(early_boundary_bic, float); full = np.asarray(full_boundary_bic, float)
    if curves.ndim != 2 or actions.shape[0] != len(curves) or early.shape != (len(curves),):
        raise ValueError("curve/action/boundary inputs do not align")
    boundary = (early > 0) & (full > 0)
    commitment = np.asarray([
        commitment_delta_bic(curve, action, horizon, immediate_hold)[0]
        for curve, action in zip(curves, actions)])
    persistence = np.asarray([
        persistence_delta_bic(curve[horizon:], late_hold) for curve in curves])
    consequential = boundary & (commitment > 0) & (persistence > 0)
    return {"alarm": bool(np.sum(consequential) >= len(curves) // 2 + 1),
            "commitment_pairs": int(np.sum(commitment > 0)),
            "persistence_pairs": int(np.sum(persistence > 0))}


def predicted_nested_rows(actions, values):
    rows = []
    for control, output in zip(actions, values):
        early, full, curves = output[:, 0], output[:, 1], output[:, 2:]
        boundary = boundary_refinement_alarm(early[None], full[None])
        action_difference = (control[-1, 0, :8] - control[-1, 1, :8])[None]
        final = consequence_from_curves(
            curves[-1:], action_difference, boundary.early_delta_bic, boundary.full_delta_bic)
        rows.append({"boundary": bool(boundary.alarm),
                     "commitment": bool(final["commitment_pairs"] >= 1),
                     "persistence": bool(final["persistence_pairs"] >= 1),
                     "alarm": bool(final["alarm"])})
    return rows


def invariance(model, start, actions, norms, device):
    model.eval(); state = torch.as_tensor(start[:2], device=device)
    control = torch.as_tensor(actions[:2], device=device)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(state, control, *norms[:4])
        swapped = model(state, control.flip(2), *norms[:4])
        reordered = model(state, control[:, permutation], *norms[:4])[:, inverse]
    return {"endpoint_swap_max_abs": float(torch.max(torch.abs(original - swapped))),
            "level_permutation_max_abs": float(torch.max(torch.abs(original - reordered)))}


def gates(errors, nested, neighborhood, checks):
    curve_checks = {
        "early_gap": errors["early_gap"] <= 1e-4,
        "full_gap": errors["full_gap"] <= 1e-4,
        "curve_action": errors["curve_action"] <= 1e-3,
        "curve_early_hold": errors["curve_early_hold"] <= 1e-3,
        "curve_late_hold": errors["curve_late_hold"] <= 1e-3,
        "nested_boundary": nested["boundary"]["agreement"] == 1.,
        "nested_final": nested["alarm"]["agreement"] == 1.,
        "candidate_not_degraded": neighborhood["candidate"]["agreement"] >= .875,
        "partition_not_degraded": neighborhood["candidate"]["mean_partition_agreement"] >= .875,
        "endpoint_swap": checks["endpoint_swap_max_abs"] <= 1e-5,
        "level_permutation": checks["level_permutation_max_abs"] <= 1e-5,
    }
    complete = dict(curve_checks)
    complete["candidate_exact"] = neighborhood["candidate"]["agreement"] == 1.
    complete["partition_exact"] = neighborhood["candidate"]["mean_partition_agreement"] == 1.
    return {"curve_formulation": {"checks": curve_checks, "passes": all(curve_checks.values())},
            "complete_same_example": {"checks": complete, "passes": all(complete.values())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    loaded = np.load(args.data, allow_pickle=False)
    data = {name: loaded[name] for name in ("start", "actions", "traces", "episode_id")}
    episodes = protocol["data"]["selected_episodes"]
    rows = selected_rows(data["episode_id"], episodes)
    nested = selected_nested(args.nested, episodes)
    val_rows, _ = validation_rows(data["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(data["start"])), val_rows)
    normalizer_np = normalizers(data["start"][train_rows], data["actions"][train_rows],
                                data["traces"][train_rows])
    norms = tensors(normalizer_np, device)
    full_scale = torch.ones(61, device=device); full_scale[:45] = norms[4]

    saved = torch.load(ROOT / protocol["model"]["backbone_checkpoint"],
                       map_location=device, weights_only=False)
    backbone = DirectSetResponseModel(set_conditioned=True).to(device)
    backbone.load_state_dict(saved["model"])
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model = NestedPairCurveHead(backbone).to(device)
    torch.manual_seed(protocol["optimization"]["seed"])
    # Reinitialize only the new head after fixing the seed; the backbone remains exactly frozen.
    model.head.reset_parameters() if hasattr(model.head, "reset_parameters") else None
    # Sequential has no reset_parameters method, so initialize its linear layers explicitly.
    for module in model.head.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()

    start = torch.as_tensor(nested[0], device=device)
    actions = torch.as_tensor(nested[1], device=device)
    target_np = curve_targets(nested[2], normalizer_np[4])
    target = torch.as_tensor(target_np, device=device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=protocol["optimization"]["learning_rate"])
    history = []; began = time.time()
    for step in range(1, protocol["optimization"]["steps"] + 1):
        model.train(); prediction = model(start, actions, *norms[:4])
        values = loss_terms(prediction, target); loss = sum(values.values())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), protocol["optimization"]["gradient_clip"])
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == protocol["optimization"]["steps"]:
            item = {"step": step, "total": float(loss.detach())}
            item.update({name: float(value.detach()) for name, value in values.items()})
            history.append(item); print(item, flush=True)
    seconds = time.time() - began
    model.eval()
    with torch.no_grad():
        prediction = model(start, actions, *norms[:4])
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        physical_values = np.expm1(target_np)
        predicted_values = decoded(prediction).cpu().numpy()
    reference_rows = nested_stage_rows(nested[1], nested[2], normalizer_np[4])
    predicted_rows = predicted_nested_rows(nested[1], predicted_values)
    nested_metrics = compare_nested(reference_rows, predicted_rows)
    neighborhood = evaluate(backbone, data, rows, nested, norms, full_scale, device)
    checks = invariance(model, nested[0], nested[1], norms, device)
    gate = gates(errors, nested_metrics, neighborhood, checks)

    checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": {"curve_steps": 39, "hidden": 128},
                "normalizers": [torch.from_numpy(value) for value in normalizer_np],
                "protocol_sha256": sha256(args.protocol)}, checkpoint)
    aggregate = json.loads((ROOT / "results/jenga/d6_staged_overfit_result.json").read_text())[
        "arms"]["set_conditioned"]["stages_by_name"]["add_nested"]
    result = {
        "protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
        "history": history, "seconds": seconds, "curve_errors": errors,
        "physical_rows": reference_rows, "predicted_rows": predicted_rows,
        "nested": nested_metrics, "neighborhood": neighborhood, "invariance": checks,
        "gates": gate,
        "aggregate_baseline": {"nested": aggregate["evaluation"]["nested"],
                               "candidate": aggregate["evaluation"]["candidate"],
                               "continuous": aggregate["evaluation"]["continuous"]},
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
        "interpretation": ("explicit_curve_passes_nested_but_complete_candidate_remains"
                           if gate["curve_formulation"]["passes"] and not gate["complete_same_example"]["passes"]
                           else "explicit_curve_complete_pass" if gate["complete_same_example"]["passes"]
                           else "explicit_curve_fails_nested_memorization"),
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"curve_errors": errors, "nested": nested_metrics,
                      "invariance": checks, "gates": gate,
                      "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
