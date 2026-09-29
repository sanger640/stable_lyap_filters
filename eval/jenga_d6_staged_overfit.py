"""Frozen staged overfit audit for localizing the failed D6 formulation."""
import argparse
import copy
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

from boundary_scale_loss import boundary_scale_terms
from jenga_d6_memorization import response_spread, selected_nested, selected_rows
from jenga_d6_set_response import (ARMS, DATA, NESTED, evaluate, normalizers, sha256, tensors)
from jenga_reobservation_interface import validation_rows
from neighborhood_topology_loss import commitment_loss, topology_loss
from set_response_model import DirectSetResponseModel, reconstruct_states


PROTOCOL = ROOT / "results/jenga/d6_staged_overfit_protocol.json"
PILOT = ROOT / "results/jenga/d6_set_response_pilot.json"
REPORT = ROOT / "results/jenga/d6_staged_overfit_result.json"
OUTPUT_DIR = ROOT / "results/jenga/d6_staged_overfit"


def objective_terms(model, data, rows, nested, norms, full_scale, active, device):
    start = torch.as_tensor(data["start"][rows], device=device)
    actions = torch.as_tensor(data["actions"][rows], device=device)
    truth = torch.as_tensor(data["traces"][rows], device=device)
    predicted_z = model(start, actions, *norms[:4])
    target_z = ((truth[..., :45] - start[:, None, None, :45]) / norms[4])
    values = {"response": (predicted_z - target_z).square().mean()}
    if "topology" in active or "commitment" in active:
        predicted = reconstruct_states(predicted_z, start, truth, norms[4])
        if "topology" in active:
            values["topology"] = topology_loss(predicted, truth, full_scale)
        if "commitment" in active:
            values["commitment"] = commitment_loss(predicted, truth, actions, full_scale)
    if "nested" in active:
        n_start = torch.as_tensor(nested[0], device=device)
        n_actions = torch.as_tensor(nested[1], device=device)
        n_truth = torch.as_tensor(nested[2], device=device)
        groups = len(n_start)
        flat_actions = n_actions.reshape(groups, -1, n_actions.shape[-2], 4)
        flat_truth = n_truth.reshape(groups, -1, n_truth.shape[-2], 61)
        nested_z = model(n_start, flat_actions, *norms[:4])
        nested_predicted = reconstruct_states(nested_z, n_start, flat_truth, norms[4])
        values["nested"] = boundary_scale_terms(
            nested_predicted.reshape_as(n_truth), n_truth, full_scale).total()
    return values


def gradient_diagnostics(model, data, rows, nested, norms, full_scale, active, weights, device):
    model.train()
    values = objective_terms(model, data, rows, nested, norms, full_scale, active, device)
    gradients = {}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = list(values)
    for index, name in enumerate(names):
        grads = torch.autograd.grad(values[name], parameters, retain_graph=index + 1 < len(names),
                                    allow_unused=True)
        gradients[name] = tuple(grads)
    norms_out = {}
    for name, grads in gradients.items():
        squared = sum((grad.detach().double().square().sum() for grad in grads if grad is not None),
                      torch.zeros((), device=device, dtype=torch.float64))
        value = float(torch.sqrt(squared).cpu())
        norms_out[name] = {"unweighted": value, "weighted": value * weights[name]}
    cosines = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            dot = torch.zeros((), device=device, dtype=torch.float64)
            for a, b in zip(gradients[left], gradients[right]):
                if a is not None and b is not None:
                    dot += (a.detach().double() * b.detach().double()).sum()
            denominator = norms_out[left]["unweighted"] * norms_out[right]["unweighted"]
            cosines[f"{left}:{right}"] = float(dot.cpu()) / max(denominator, 1e-20)
    return {"losses": {name: float(value.detach()) for name, value in values.items()},
            "gradient_norms": norms_out, "gradient_cosines": cosines}


def train_stage(model, data, rows, nested, norms, full_scale, stage, weights, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    active = tuple(stage["terms"]); history = []; began = time.time()
    for step in range(1, stage["steps"] + 1):
        model.train()
        values = objective_terms(model, data, rows, nested, norms, full_scale, active, device)
        loss = sum(weights[name] * values[name] for name in active)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == stage["steps"]:
            record = {"step": step, "total": float(loss.detach())}
            record.update({name: float(value.detach()) for name, value in values.items()})
            history.append(record)
            print(f"{stage['name']} {step}/{stage['steps']}: {record}", flush=True)
    return history, time.time() - began


def stage_gate(name, evaluation, spread):
    continuous = evaluation["continuous"]
    checks = {"response": continuous["response"] <= (0.005 if name == "one_state_response" else 0.02)}
    if name in ("add_topology", "add_commitment", "add_nested"):
        checks["topology"] = continuous["topology"] <= 0.01
    if name in ("add_commitment", "add_nested"):
        checks["commitment"] = continuous["commitment"] <= 0.01
    if name == "add_nested":
        checks.update({
            "candidate_decision": evaluation["candidate"]["agreement"] == 1.0,
            "partition": evaluation["candidate"]["mean_partition_agreement"] >= 0.90,
            "nested_boundary": evaluation["nested"]["boundary"]["agreement"] >= 0.90,
            "nested_final": evaluation["nested"]["alarm"]["agreement"] >= 0.90,
            "response_spread": all(0.5 <= value <= 1.5 for value in spread.values()),
            "permutation": evaluation["checks"]["permutation_max_abs"] <= 1e-5,
        })
    return {"checks": checks, "passes": all(checks.values())}


def diagnosis(results):
    order = [stage["name"] for stage in results[ARMS[0]]["stages"]]
    first_failures = {}
    for arm in ARMS:
        first_failures[arm] = next(
            (name for name in order if not results[arm]["stages_by_name"][name]["gate"]["passes"]),
            None)
    set_stages = results["set_conditioned"]["stages_by_name"]
    if (set_stages["add_commitment"]["gate"]["passes"]
            and not set_stages["add_nested"]["gate"]["passes"]):
        conclusion = "neighborhood_geometry_representable_nested_consequence_bottleneck"
    elif all(value is None for value in first_failures.values()):
        conclusion = "all_stages_pass_generalization_or_data_next"
    elif all(value == "one_state_response" for value in first_failures.values()):
        conclusion = "one_state_response_capacity_or_optimization_failure"
    elif any(value == "eight_state_response" for value in first_failures.values()):
        conclusion = "multi_scene_conditioning_or_capacity_failure"
    else:
        conclusion = "relational_objective_interference"
    return {"first_failed_stage": first_failures, "conclusion": conclusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument("--pilot", default=str(PILOT))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = Path(args.report)
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
    weights = json.loads(Path(args.pilot).read_text())["loss_weights"]
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(protocol["models"]["initialization_seed"])
    independent = DirectSetResponseModel(set_conditioned=False)
    independent_initial = copy.deepcopy(independent.state_dict())
    conditioned = DirectSetResponseModel(set_conditioned=True)
    conditioned_initial = copy.deepcopy(conditioned.state_dict())
    for key in conditioned_initial:
        if key in independent_initial and conditioned_initial[key].shape == independent_initial[key].shape:
            conditioned_initial[key] = independent_initial[key].clone()

    arms = {}
    for arm, initial in zip(ARMS, (independent_initial, conditioned_initial)):
        model = DirectSetResponseModel(set_conditioned=arm == "set_conditioned").to(device)
        model.load_state_dict(initial)
        stage_results = []
        for stage in protocol["optimization"]["stages"]:
            stage_rows = rows[:1] if stage["states"] == 1 else rows
            stage_nested = tuple(value[:1] for value in nested) if stage["states"] == 1 else nested
            print(f"{arm}: starting {stage['name']}", flush=True)
            before = gradient_diagnostics(model, data, stage_rows, stage_nested, norms, full_scale,
                                          stage["terms"], weights, device)
            history, seconds = train_stage(model, data, stage_rows, stage_nested, norms, full_scale,
                                           stage, weights, protocol["optimization"]["learning_rate"], device)
            after = gradient_diagnostics(model, data, stage_rows, stage_nested, norms, full_scale,
                                         stage["terms"], weights, device)
            evaluation = evaluate(model, data, stage_rows, stage_nested, norms, full_scale, device)
            spread = response_spread(model, data, stage_rows, norms, device)
            gate = stage_gate(stage["name"], evaluation, spread)
            checkpoint = output_dir / f"{arm}_{stage['name']}.pt"
            torch.save({"model": model.state_dict(), "arm": arm, "stage": stage["name"],
                        "normalizers": [torch.from_numpy(value) for value in normalizer_np],
                        "protocol_sha256": sha256(args.protocol)}, checkpoint)
            item = {"name": stage["name"], "terms": stage["terms"], "steps": stage["steps"],
                    "history": history, "seconds": seconds, "gradient_before": before,
                    "gradient_after": after, "evaluation": evaluation,
                    "spread_ratio": spread, "gate": gate,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint)}
            stage_results.append(item)
            print(json.dumps({arm: {stage["name"]: {"evaluation": evaluation,
                  "spread": spread, "gate": gate}}}, indent=2), flush=True)
        arms[arm] = {"stages": stage_results,
                     "stages_by_name": {item["name"]: item for item in stage_results}}
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    result = {"protocol": protocol, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "protocol_sha256": sha256(args.protocol), "data_sha256": sha256(args.data),
              "pilot_sha256": sha256(args.pilot), "loss_weights": weights,
              "selected_rows": rows.tolist(), "arms": arms}
    result["diagnosis"] = diagnosis(arms)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["diagnosis"], indent=2), flush=True)


if __name__ == "__main__":
    main()
