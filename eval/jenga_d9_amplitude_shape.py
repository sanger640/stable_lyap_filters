"""Frozen D9 exact-zero amplitude-times-shape confirmation."""
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

from jenga_d6_memorization import selected_nested
from jenga_d6_set_response import DATA, NESTED, normalizers, sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded, loss_terms, predicted_nested_rows
from jenga_reobservation_interface import compare_nested, nested_stage_rows, validation_rows
from set_response_model import AmplitudeShapeTemporalCurveModel


PROTOCOL = ROOT / "results/jenga/d9_amplitude_shape_protocol.json"
REPORT = ROOT / "results/jenga/d9_amplitude_shape_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d9_amplitude_shape"
D8B_RESULT = ROOT / "results/jenga/d8b_stability_result.json"


def factor_targets(target):
    amplitude = torch.sqrt(torch.mean(target.square(), dim=-1, keepdim=True))
    nonzero = amplitude > 0
    shape = torch.where(nonzero, target / amplitude.clamp_min(1e-12), torch.zeros_like(target))
    return amplitude, shape, nonzero


def masked_mse(predicted, target, mask, section):
    selected = mask.expand_as(target)[..., section]
    differences = (predicted[..., section] - target[..., section]).square()
    return differences[selected].mean() if torch.any(selected) else differences.sum() * 0.


def factor_losses(predicted_amplitude, predicted_shape, target_amplitude, target_shape, nonzero):
    return {
        "amplitude": (predicted_amplitude - target_amplitude).square().mean(),
        "shape_early_gap": masked_mse(predicted_shape, target_shape, nonzero, slice(0, 1)),
        "shape_full_gap": masked_mse(predicted_shape, target_shape, nonzero, slice(1, 2)),
        "shape_action": masked_mse(predicted_shape, target_shape, nonzero, slice(2, 11)),
        "shape_early_hold": masked_mse(predicted_shape, target_shape, nonzero, slice(11, 21)),
        "shape_late_hold": masked_mse(predicted_shape, target_shape, nonzero, slice(21, 41)),
    }


def invariance(model, start, actions, norms, device):
    model.eval(); state = torch.as_tensor(start[:2], device=device)
    control = torch.as_tensor(actions[:2], device=device)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2], device=device); inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(state, control, *norms[:4])[0]
        swapped = model(state, control.flip(2), *norms[:4])[0]
        reordered = model(state, control[:, permutation], *norms[:4])[0][:, inverse]
    return {"endpoint_swap_max_abs": float(torch.max(torch.abs(original - swapped))),
            "level_permutation_max_abs": float(torch.max(torch.abs(original - reordered)))}


def gate(errors, nested, checks, predicted_amplitude, target_amplitude):
    zero = target_amplitude == 0; nonzero = target_amplitude > 0
    conditions = {
        "early_gap": errors["early_gap"] <= 1e-4,
        "full_gap": errors["full_gap"] <= 1e-4,
        "curve_action": errors["curve_action"] <= 1e-3,
        "curve_early_hold": errors["curve_early_hold"] <= 1e-3,
        "curve_late_hold": errors["curve_late_hold"] <= 1e-3,
        "exact_zero": bool(torch.all(predicted_amplitude[zero] == 0)),
        "nonzero_positive": bool(torch.all(predicted_amplitude[nonzero] > 0)),
        "nested_boundary": nested["boundary"]["agreement"] == 1.,
        "nested_final": nested["alarm"]["agreement"] == 1.,
        "endpoint_swap": checks["endpoint_swap_max_abs"] <= 1e-5,
        "level_permutation": checks["level_permutation_max_abs"] <= 1e-5,
    }
    return {"checks": conditions, "passes": all(conditions.values()),
            "zero_levels": int(zero.sum()),
            "predicted_zero_levels": int((predicted_amplitude == 0).sum())}


def train_seed(seed, protocol, start, actions, target, target_amplitude, target_shape, nonzero,
               nested, target_scale, norms, device, output_dir):
    torch.manual_seed(seed)
    model = AmplitudeShapeTemporalCurveModel(
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"]).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history = []; began = time.time()
    for step in range(1, optimization["steps"] + 1):
        model.train(); prediction, amplitude, shape = model(start, actions, *norms[:4])
        values = factor_losses(amplitude, shape, target_amplitude, target_shape, nonzero)
        loss = sum(values.values())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 500 == 0 or step == optimization["steps"]:
            item = {"step": step, "learning_rate": scheduler.get_last_lr()[0],
                    "total": float(loss.detach())}
            item.update({name: float(value.detach()) for name, value in values.items()})
            history.append(item); print(f"seed {seed}: {item}", flush=True)
    seconds = time.time() - began
    model.eval()
    with torch.no_grad():
        prediction, amplitude, shape = model(start, actions, *norms[:4])
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        factor = {name: float(value) for name, value in factor_losses(
            amplitude, shape, target_amplitude, target_shape, nonzero).items()}
        predicted_values = decoded(prediction).cpu().numpy()
    reference = nested_stage_rows(nested[1], nested[2], target_scale)
    predicted = predicted_nested_rows(nested[1], predicted_values)
    nested_metrics = compare_nested(reference, predicted)
    checks = invariance(model, nested[0], nested[1], norms, device)
    result_gate = gate(errors, nested_metrics, checks, amplitude, target_amplitude)
    checkpoint = output_dir / f"seed{seed}.pt"
    torch.save({"model": model.state_dict(), "model_config": protocol["model"], "seed": seed,
                "normalizers": [value.detach().cpu() for value in norms],
                "protocol_sha256": sha256(PROTOCOL)}, checkpoint)
    return {"seed": seed, "seconds": seconds, "history": history, "curve_errors": errors,
            "factor_losses": factor, "nested": nested_metrics, "invariance": checks,
            "gate": result_gate, "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint)}


def aggregate(seeds):
    return {"all_seeds_pass": all(row["gate"]["passes"] for row in seeds),
            "passing_seeds": sum(row["gate"]["passes"] for row in seeds),
            "boundary_agreement": [row["nested"]["boundary"]["agreement"] for row in seeds],
            "final_agreement": [row["nested"]["alarm"]["agreement"] for row in seeds],
            "predicted_zero_levels": [row["gate"]["predicted_zero_levels"] for row in seeds]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR)); parser.add_argument("--d8b-result", default=str(D8B_RESULT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text())
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    loaded = np.load(args.data, allow_pickle=False)
    val_rows, _ = validation_rows(loaded["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(loaded["start"])), val_rows)
    normalizer_np = normalizers(loaded["start"][train_rows], loaded["actions"][train_rows],
                                loaded["traces"][train_rows]); norms = tensors(normalizer_np, device)
    nested = selected_nested(args.nested, protocol["data"]["selected_episodes"])
    start = torch.as_tensor(nested[0], device=device); actions = torch.as_tensor(nested[1], device=device)
    target_np = curve_targets(nested[2], normalizer_np[4]); target = torch.as_tensor(target_np, device=device)
    target_amplitude, target_shape, nonzero = factor_targets(target)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [train_seed(seed, protocol, start, actions, target, target_amplitude, target_shape,
                        nonzero, nested, normalizer_np[4], norms, device, output_dir)
             for seed in protocol["optimization"]["seeds"]]
    summary = aggregate(seeds)
    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
              "target_zero_levels": int((target_amplitude == 0).sum()), "seeds": seeds,
              "aggregate": summary,
              "d8b": {"result_sha256": sha256(args.d8b_result),
                       "aggregate": json.loads(Path(args.d8b_result).read_text())["aggregate"]},
              "interpretation": ("all_seed_pass_authorize_episode_heldout_train"
                                 if summary["all_seeds_pass"]
                                 else "zero_preserving_confirmation_failed")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"aggregate": summary, "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
