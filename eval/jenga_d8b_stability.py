"""Frozen three-seed stability confirmation for D8 raw temporal curves."""
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
from jenga_d8_raw_temporal_curve import frozen_gate, invariance
from jenga_reobservation_interface import compare_nested, nested_stage_rows, validation_rows
from set_response_model import DirectTemporalPairCurveModel


PROTOCOL = ROOT / "results/jenga/d8b_stability_protocol.json"
REPORT = ROOT / "results/jenga/d8b_stability_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d8b_stability"
D8_RESULT = ROOT / "results/jenga/d8_raw_temporal_curve_result.json"


def train_seed(seed, protocol, start, actions, target, nested, target_scale, norms, device,
               output_dir):
    torch.manual_seed(seed)
    model = DirectTemporalPairCurveModel(
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"], action_horizon=protocol["model"]["action_horizon"],
        curve_steps=protocol["model"]["curve_steps"]).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history = []; began = time.time()
    for step in range(1, optimization["steps"] + 1):
        model.train(); prediction = model(start, actions, *norms[:4])
        values = loss_terms(prediction, target); loss = sum(values.values())
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
        prediction = model(start, actions, *norms[:4])
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        predicted_values = decoded(prediction).cpu().numpy()
    reference = nested_stage_rows(nested[1], nested[2], target_scale)
    predicted = predicted_nested_rows(nested[1], predicted_values)
    nested_metrics = compare_nested(reference, predicted)
    checks = invariance(model, nested[0], nested[1], norms, device)
    gate = frozen_gate(errors, nested_metrics, checks)
    checkpoint = output_dir / f"seed{seed}.pt"
    torch.save({"model": model.state_dict(), "model_config": protocol["model"], "seed": seed,
                "normalizers": [value.detach().cpu() for value in norms],
                "protocol_sha256": sha256(PROTOCOL)}, checkpoint)
    return {"seed": seed, "seconds": seconds, "history": history, "curve_errors": errors,
            "nested": nested_metrics, "invariance": checks, "gate": gate,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint)}


def aggregate(seeds):
    fields = ("early_gap", "full_gap", "curve_action", "curve_early_hold", "curve_late_hold")
    return {
        "all_seeds_pass": all(row["gate"]["passes"] for row in seeds),
        "passing_seeds": sum(row["gate"]["passes"] for row in seeds),
        "curve_error_ranges": {name: {
            "min": min(row["curve_errors"][name] for row in seeds),
            "median": float(np.median([row["curve_errors"][name] for row in seeds])),
            "max": max(row["curve_errors"][name] for row in seeds),
        } for name in fields},
        "boundary_agreement": [row["nested"]["boundary"]["agreement"] for row in seeds],
        "final_agreement": [row["nested"]["alarm"]["agreement"] for row in seeds],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR)); parser.add_argument("--d8-result", default=str(D8_RESULT))
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
    nested = selected_nested(args.nested, protocol["data"]["selected_episodes"])
    start = torch.as_tensor(nested[0], device=device); actions = torch.as_tensor(nested[1], device=device)
    target_np = curve_targets(nested[2], normalizer_np[4]); target = torch.as_tensor(target_np, device=device)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [train_seed(seed, protocol, start, actions, target, nested, normalizer_np[4], norms,
                        device, output_dir) for seed in protocol["optimization"]["seeds"]]
    summary = aggregate(seeds)
    result = {
        "protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
        "seeds": seeds, "aggregate": summary,
        "constant_lr_d8": json.loads(Path(args.d8_result).read_text()),
        "interpretation": ("stable_pass_authorize_episode_heldout_train" if summary["all_seeds_pass"]
                           else "stability_confirmation_failed_keep_heldout_closed"),
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"aggregate": summary, "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
