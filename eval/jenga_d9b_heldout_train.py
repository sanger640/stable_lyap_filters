"""Frozen episode-held-out TRAIN evaluation of the D9b direct response model."""
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

from jenga_d6_set_response import DATA, NESTED, normalizers, sha256, tensors
from jenga_d7_explicit_curve import curve_targets, decoded, loss_terms, predicted_nested_rows
from jenga_d9_amplitude_shape import invariance
from jenga_d9b_shared_amplitude import factor_losses, factor_targets
from jenga_reobservation_interface import (binary_metrics, compare_nested, load_nested,
                                           nested_stage_rows, validation_rows)
from set_response_model import GroupAmplitudeShapeTemporalCurveModel


PROTOCOL = ROOT / "results/jenga/d9b_heldout_train_protocol.json"
REPORT = ROOT / "results/jenga/d9b_heldout_train_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d9b_heldout_train"
D2_RESULT = ROOT / "results/jenga/reobservation_interface_result.json"


def zero_metrics(reference_amplitude, predicted_amplitude):
    reference = np.asarray(reference_amplitude) == 0
    predicted = np.asarray(predicted_amplitude) == 0
    tp = int(np.sum(reference & predicted)); fn = int(np.sum(reference & ~predicted))
    fp = int(np.sum(~reference & predicted)); tn = int(np.sum(~reference & ~predicted))
    return {"groups": int(len(reference)), "reference_zero": int(reference.sum()),
            "predicted_zero": int(predicted.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": float(tp / max(tp + fn, 1)),
            "false_zero_rate": float(fp / max(fp + tn, 1))}


def median_summary(rows):
    summary = {}
    for stage in ("boundary", "commitment", "persistence", "alarm"):
        summary[stage] = {field: float(np.median([
            row["nested"][stage][field] for row in rows]))
            for field in ("agreement", "recall", "added_positive_rate")}
    summary["zero"] = {field: float(np.median([row["zero"][field] for row in rows]))
                       for field in ("recall", "false_zero_rate")}
    return summary


def success_gate(summary, rows, d2):
    comparisons = {
        "boundary_recall_gain": summary["boundary"]["recall"] - d2["boundary"]["recall"],
        "boundary_added_positive_increase": (summary["boundary"]["added_positive_rate"]
                                             - d2["boundary"]["added_positive_rate"]),
        "final_recall_gain": summary["alarm"]["recall"] - d2["alarm"]["recall"],
        "final_added_positive_increase": (summary["alarm"]["added_positive_rate"]
                                          - d2["alarm"]["added_positive_rate"]),
    }
    checks = {
        "boundary_recall": comparisons["boundary_recall_gain"] >= .10,
        "boundary_added_positive": comparisons["boundary_added_positive_increase"] <= .05,
        "final_recall": comparisons["final_recall_gain"] >= .15,
        "final_added_positive": comparisons["final_added_positive_increase"] <= .05,
        "endpoint_swap": all(row["invariance"]["endpoint_swap_max_abs"] <= 1e-5 for row in rows),
        "level_permutation": all(
            row["invariance"]["level_permutation_max_abs"] <= 1e-5 for row in rows),
    }
    return {"comparisons": comparisons, "checks": checks, "passes": all(checks.values())}


def evaluate(model, start, actions, target, truth, sources, target_amplitude, norms,
             response_scale, device):
    model.eval()
    with torch.no_grad():
        prediction, amplitude, shape = model(start, actions, *norms[:4])
        errors = {name: float(value) for name, value in loss_terms(prediction, target).items()}
        values = decoded(prediction).cpu().numpy()
        predicted_amplitude = amplitude[:, 0, 0].cpu().numpy()
    reference_rows = nested_stage_rows(actions.cpu().numpy(), truth, response_scale)
    predicted_rows = predicted_nested_rows(actions.cpu().numpy(), values)
    nested = compare_nested(reference_rows, predicted_rows)
    zero = zero_metrics(target_amplitude[:, 0, 0].cpu().numpy(), predicted_amplitude)
    evidence = []
    for index, (source, reference, predicted) in enumerate(
            zip(sources, reference_rows, predicted_rows)):
        evidence.append({"index": index, "source": source,
                         "episode": source.split("_seed")[0][2:],
                         "target_amplitude": float(target_amplitude[index, 0, 0]),
                         "predicted_amplitude": float(predicted_amplitude[index]),
                         "reference": reference, "predicted": predicted})
    checks = invariance(model, start.cpu().numpy(), actions.cpu().numpy(), norms, device)
    return {"curve_errors": errors, "nested": nested, "zero": zero,
            "invariance": checks, "evidence": evidence}


def train_seed(seed, protocol, train, heldout, norms, response_scale, device, output_dir):
    train_start, train_actions, train_target, train_amplitude, train_shape, train_nonzero = train
    held_start, held_actions, held_target, held_truth, held_sources, held_amplitude = heldout
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GroupAmplitudeShapeTemporalCurveModel(
        hidden=protocol["model"]["hidden"], heads=protocol["model"]["heads"],
        layers=protocol["model"]["layers"]).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["steps"], eta_min=optimization["final_learning_rate"])
    history = []; began = time.time(); batch = optimization["group_batch_size"]
    for step in range(1, optimization["steps"] + 1):
        indices = torch.as_tensor(rng.choice(len(train_start), batch, replace=False), device=device)
        model.train(); prediction, amplitude, shape = model(
            train_start[indices], train_actions[indices], *norms[:4])
        losses = factor_losses(amplitude, shape, train_amplitude[indices],
                               train_shape[indices], train_nonzero[indices])
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
        optimizer.step(); scheduler.step()
        if step == 1 or step % 500 == 0 or step == optimization["steps"]:
            item = {"step": step, "learning_rate": scheduler.get_last_lr()[0],
                    "total": float(loss.detach())}
            item.update({name: float(value.detach()) for name, value in losses.items()})
            history.append(item); print(f"seed {seed}: {item}", flush=True)
    evaluation = evaluate(model, held_start, held_actions, held_target, held_truth,
                          held_sources, held_amplitude, norms, response_scale, device)
    checkpoint = output_dir / f"seed{seed}.pt"
    torch.save({"model": model.state_dict(), "model_config": protocol["model"], "seed": seed,
                "normalizers": [value.detach().cpu() for value in norms],
                "protocol_sha256": sha256(PROTOCOL)}, checkpoint)
    evaluation.update({"seed": seed, "seconds": time.time() - began, "history": history,
                       "checkpoint": str(checkpoint.relative_to(ROOT)),
                       "checkpoint_sha256": sha256(checkpoint)})
    return evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR)); parser.add_argument("--d2-result", default=str(D2_RESULT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = Path(args.report)
    if report.exists() and not args.force: raise SystemExit(f"refusing to overwrite {report}")
    protocol = json.loads(Path(args.protocol).read_text())
    if sha256(args.data) != protocol["data"]["neighborhood_sha256"]:
        raise SystemExit("neighborhood data hash differs from frozen protocol")
    if sha256(args.d2_result) != protocol["d2_reference"]["source_sha256"]:
        raise SystemExit("D2 reference hash differs from frozen protocol")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = np.load(args.data, allow_pickle=False)
    val_rows, heldout_episodes = validation_rows(source["episode_id"])
    heldout_episodes = list(map(str, heldout_episodes))
    if heldout_episodes != protocol["data"]["heldout_episode_ids"]:
        raise SystemExit("held-out episodes differ from frozen protocol")
    train_rows = np.setdiff1d(np.arange(len(source["start"])), val_rows)
    normalizer_np = normalizers(source["start"][train_rows], source["actions"][train_rows],
                                source["traces"][train_rows]); norms = tensors(normalizer_np, device)
    all_episodes = sorted(set(source["episode_id"].astype(str)), key=int)
    fit_episodes = [episode for episode in all_episodes if episode not in set(heldout_episodes)]
    fit_start_np, fit_actions_np, fit_truth, _ = load_nested(args.nested, fit_episodes)
    held_start_np, held_actions_np, held_truth, held_sources = load_nested(
        args.nested, heldout_episodes)
    if len(fit_start_np) != protocol["data"]["fit_groups"] or len(held_start_np) != protocol["data"]["heldout_groups"]:
        raise SystemExit("nested group counts differ from frozen protocol")

    fit_target = torch.as_tensor(curve_targets(fit_truth, normalizer_np[4]), device=device)
    fit_amplitude, fit_shape, fit_nonzero = factor_targets(fit_target)
    held_target = torch.as_tensor(curve_targets(held_truth, normalizer_np[4]), device=device)
    held_amplitude, _, _ = factor_targets(held_target)
    train = (torch.as_tensor(fit_start_np, device=device),
             torch.as_tensor(fit_actions_np, device=device), fit_target,
             fit_amplitude, fit_shape, fit_nonzero)
    heldout = (torch.as_tensor(held_start_np, device=device),
               torch.as_tensor(held_actions_np, device=device), held_target,
               held_truth, held_sources, held_amplitude)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [train_seed(seed, protocol, train, heldout, norms, normalizer_np[4], device, output_dir)
             for seed in protocol["optimization"]["seeds"]]

    d2_result = json.loads(Path(args.d2_result).read_text())
    d2 = {stage: {field: d2_result["summary"]["38"]["nested"][stage][field]["median"]
                  for field in ("agreement", "recall", "added_positive_rate")}
          for stage in ("boundary", "commitment", "persistence", "alarm")}
    summary = median_summary(seeds); gate = success_gate(summary, seeds, d2)
    result = {"protocol": protocol,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
              "heldout_episodes": heldout_episodes, "fit_groups": len(fit_start_np),
              "heldout_groups": len(held_start_np), "seeds": seeds,
              "median": summary, "d2_median": d2, "success_gate": gate,
              "interpretation": ("pass_authorize_frozen_dev_protocol" if gate["passes"]
                                 else "fail_keep_dev_test_closed")}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median": summary, "d2_median": d2, "success_gate": gate,
                      "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
