"""Matched D8 raw-action temporal curve capacity audit."""
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
from set_response_model import DirectTemporalPairCurveModel


PROTOCOL = ROOT / "results/jenga/d8_raw_temporal_curve_protocol.json"
REPORT = ROOT / "results/jenga/d8_raw_temporal_curve_result.json"
CHECKPOINT = ROOT / "results/jenga/d8_raw_temporal_curve/model.pt"
D7_RESULT = ROOT / "results/jenga/d7_explicit_curve_result.json"


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


def frozen_gate(errors, nested, checks):
    conditions = {
        "early_gap": errors["early_gap"] <= 1e-4,
        "full_gap": errors["full_gap"] <= 1e-4,
        "curve_action": errors["curve_action"] <= 1e-3,
        "curve_early_hold": errors["curve_early_hold"] <= 1e-3,
        "curve_late_hold": errors["curve_late_hold"] <= 1e-3,
        "nested_boundary": nested["boundary"]["agreement"] == 1.,
        "nested_final": nested["alarm"]["agreement"] == 1.,
        "endpoint_swap": checks["endpoint_swap_max_abs"] <= 1e-5,
        "level_permutation": checks["level_permutation_max_abs"] <= 1e-5,
    }
    return {"checks": conditions, "passes": all(conditions.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT)); parser.add_argument("--d7-result", default=str(D7_RESULT))
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
                                loaded["traces"][train_rows])
    norms = tensors(normalizer_np, device)
    episodes = protocol["data"]["selected_episodes"]
    nested = selected_nested(args.nested, episodes)
    start = torch.as_tensor(nested[0], device=device)
    actions = torch.as_tensor(nested[1], device=device)
    target_np = curve_targets(nested[2], normalizer_np[4])
    target = torch.as_tensor(target_np, device=device)

    torch.manual_seed(protocol["optimization"]["seed"])
    model = DirectTemporalPairCurveModel(
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["optimization"]["learning_rate"])
    history = []; began = time.time()
    for step in range(1, protocol["optimization"]["steps"] + 1):
        model.train(); prediction = model(start, actions, *norms[:4])
        values = loss_terms(prediction, target); loss = sum(values.values())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["optimization"]["gradient_clip"])
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
        predicted_values = decoded(prediction).cpu().numpy()
    reference_rows = nested_stage_rows(nested[1], nested[2], normalizer_np[4])
    predicted_rows = predicted_nested_rows(nested[1], predicted_values)
    nested_metrics = compare_nested(reference_rows, predicted_rows)
    checks = invariance(model, nested[0], nested[1], norms, device)
    gate = frozen_gate(errors, nested_metrics, checks)
    checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": protocol["model"],
                "normalizers": [torch.from_numpy(value) for value in normalizer_np],
                "protocol_sha256": sha256(args.protocol)}, checkpoint)
    d7 = json.loads(Path(args.d7_result).read_text())
    result = {
        "protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
        "history": history, "seconds": seconds, "curve_errors": errors,
        "physical_rows": reference_rows, "predicted_rows": predicted_rows,
        "nested": nested_metrics, "invariance": checks, "gate": gate,
        "d7_frozen_token": {"curve_errors": d7["curve_errors"], "nested": d7["nested"],
                            "invariance": d7["invariance"]},
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
        "interpretation": ("direct_temporal_passes_d6_tokens_are_bottleneck" if gate["passes"]
                           else "direct_temporal_fails_curve_or_optimization_bottleneck"),
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"curve_errors": errors, "nested": nested_metrics, "invariance": checks,
                      "gate": gate, "interpretation": result["interpretation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
