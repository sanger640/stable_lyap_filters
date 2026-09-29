"""Frozen TRAIN-only memorization audit for the failed D6 response models."""
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

from jenga_d6_set_response import (ARMS, DATA, NESTED, evaluate, loss_terms, normalizers,
                                   sha256, tensors)
from jenga_reobservation_interface import load_nested, validation_rows
from set_response_model import DirectSetResponseModel


PROTOCOL = ROOT / "results/jenga/d6_memorization_protocol.json"
PILOT = ROOT / "results/jenga/d6_set_response_pilot.json"
OUTPUT_DIR = ROOT / "results/jenga/d6_memorization"
REPORT = ROOT / "results/jenga/d6_memorization_result.json"


def selected_rows(episode_ids, selected_episodes):
    episodes = np.asarray(episode_ids).astype(str)
    rows = []
    for episode in selected_episodes:
        found = np.flatnonzero(episodes == str(episode))
        if not len(found):
            raise RuntimeError(f"selected episode {episode} is absent")
        rows.append(int(found[0]))
    return np.asarray(rows)


def selected_nested(directory, episodes):
    start, actions, traces, sources = load_nested(directory, episodes)
    chosen = []
    for episode in episodes:
        found = [index for index, source in enumerate(sources)
                 if source.split("_seed")[0][2:] == str(episode)]
        if not found:
            raise RuntimeError(f"selected episode {episode} has no nested group")
        chosen.append(found[0])
    chosen = np.asarray(chosen)
    return start[chosen], actions[chosen], traces[chosen]


def permuted(values, rng):
    actions, traces = values
    probes = actions.shape[1]
    index = np.stack([rng.permutation(probes) for _ in range(len(actions))])
    row = np.arange(len(actions))[:, None]
    return actions[row, index], traces[row, index]


def train(name, initial, data, rows, nested, norms, full_scale, weights, args, device):
    model = DirectSetResponseModel(
        hidden=128, heads=4, layers=2, set_conditioned=name == "set_conditioned").to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history = []; began = time.time()
    for step in range(1, args.steps + 1):
        pick = rng.choice(len(rows), args.batch, replace=False)
        selected = rows[pick]
        action_np, truth_np = permuted((data["actions"][selected], data["traces"][selected]), rng)
        nested_pick = rng.choice(len(nested[0]), args.batch, replace=False)
        values = loss_terms(
            model,
            torch.as_tensor(data["start"][selected], device=device),
            torch.as_tensor(action_np, device=device),
            torch.as_tensor(truth_np, device=device),
            torch.as_tensor(nested[0][nested_pick], device=device),
            torch.as_tensor(nested[1][nested_pick], device=device),
            torch.as_tensor(nested[2][nested_pick], device=device),
            norms, full_scale, device)
        loss = sum(weights[key] * value for key, value in values.items())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {key: float(value.detach()) for key, value in values.items()}
            row.update({"step": step, "total": float(loss.detach())})
            history.append(row); print(f"{name} {step}/{args.steps}: {row}", flush=True)
    return model, history, time.time() - began


def response_spread(model, data, rows, norms, device):
    model.eval(); output = []
    with torch.no_grad():
        for first in range(0, len(rows), 4):
            chosen = rows[first:first + 4]
            output.append(model(
                torch.as_tensor(data["start"][chosen], device=device),
                torch.as_tensor(data["actions"][chosen], device=device), *norms[:4]
            ).cpu().numpy())
    predicted = np.concatenate(output)
    start = data["start"][rows]
    truth = ((data["traces"][rows][..., :45] - start[:, None, None, :45])
             / norms[4].cpu().numpy())
    phases = {"action": slice(0, 8), "early_hold": slice(8, 18),
              "late_hold": slice(18, 38)}
    result = {}
    for name, phase in phases.items():
        p = np.sqrt(np.mean((predicted[:, :, phase]
                            - predicted[:, :, phase].mean(1, keepdims=True)) ** 2,
                           axis=(1, 2, 3)))
        q = np.sqrt(np.mean((truth[:, :, phase]
                            - truth[:, :, phase].mean(1, keepdims=True)) ** 2,
                           axis=(1, 2, 3)))
        result[name] = float(np.median(p / np.maximum(q, 1e-8)))
    return result


def capacity_gate(evaluation, spread):
    checks = {
        "response": evaluation["continuous"]["response"] <= .02,
        "topology": evaluation["continuous"]["topology"] <= .01,
        "commitment": evaluation["continuous"]["commitment"] <= .01,
        "candidate_decision": evaluation["candidate"]["agreement"] == 1.,
        "partition": evaluation["candidate"]["mean_partition_agreement"] >= .90,
        "nested_boundary": evaluation["nested"]["boundary"]["agreement"] >= .90,
        "nested_final": evaluation["nested"]["alarm"]["agreement"] >= .90,
        "response_spread": all(.5 <= value <= 1.5 for value in spread.values()),
        "permutation": evaluation["checks"]["permutation_max_abs"] <= 1e-5,
    }
    return {"checks": checks, "passes": all(checks.values())}


def interpret(gates):
    passing = [name for name, result in gates.items() if result["passes"]]
    if len(passing) == 2:
        return "both_representable_data_generalization"
    if passing == ["independent_direct"]:
        return "independent_only_set_conditioning_collapse"
    if passing == ["set_conditioned"]:
        return "set_only_joint_context_required"
    return "neither_passes_architecture_objective_optimization"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--pilot", default=str(PILOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol = json.loads(Path(args.protocol).read_text())
    frozen = protocol["optimization"]
    if (args.steps != frozen["steps"] or args.batch != frozen["states_per_step"]
            or args.lr != frozen["learning_rate"] or args.seed != 1):
        raise SystemExit("arguments differ from frozen protocol")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data_file = np.load(args.data, allow_pickle=False)
    data = {name: data_file[name] for name in ("start", "actions", "traces", "episode_id")}
    rows = selected_rows(data["episode_id"], protocol["data"]["selected_episodes"])
    nested = selected_nested(args.nested, protocol["data"]["selected_episodes"])

    val_rows, _ = validation_rows(data["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(data["start"])), val_rows)
    normalizer_np = normalizers(
        data["start"][train_rows], data["actions"][train_rows], data["traces"][train_rows])
    norms = tensors(normalizer_np, device)
    full_scale = torch.ones(61, device=device); full_scale[:45] = norms[4]
    weights = json.loads(Path(args.pilot).read_text())["loss_weights"]

    torch.manual_seed(args.seed)
    independent = DirectSetResponseModel(set_conditioned=False)
    independent_initial = copy.deepcopy(independent.state_dict())
    conditioned = DirectSetResponseModel(set_conditioned=True)
    conditioned_initial = copy.deepcopy(conditioned.state_dict())
    for key in conditioned_initial:
        if key in independent_initial and conditioned_initial[key].shape == independent_initial[key].shape:
            conditioned_initial[key] = independent_initial[key].clone()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    arms = {}; gates = {}
    for name, initial in zip(ARMS, (independent_initial, conditioned_initial)):
        torch.manual_seed(args.seed)
        model, history, seconds = train(
            name, initial, data, rows, nested, norms, full_scale, weights, args, device)
        evaluation = evaluate(model, data, rows, nested, norms, full_scale, device)
        spread = response_spread(model, data, rows, norms, device)
        gates[name] = capacity_gate(evaluation, spread)
        checkpoint = output_dir / f"{name}_s{args.seed}.pt"
        torch.save({"model": model.state_dict(), "arm": name, "seed": args.seed,
                    "model_config": {"steps": 38, "hidden": 128, "heads": 4, "layers": 2,
                                     "set_conditioned": name == "set_conditioned"},
                    "normalizers": [torch.from_numpy(value) for value in normalizer_np],
                    "protocol_sha256": sha256(args.protocol)}, checkpoint)
        arms[name] = {"history": history, "seconds": seconds, "evaluation": evaluation,
                      "spread_ratio": spread,
                      "checkpoint": str(checkpoint.relative_to(ROOT)),
                      "checkpoint_sha256": sha256(checkpoint)}
        print(json.dumps({name: {"evaluation": evaluation, "spread": spread,
                                 "gate": gates[name]}}, indent=2), flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    result = {
        "protocol": protocol,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol), "pilot_sha256": sha256(args.pilot),
        "data_sha256": sha256(args.data),
        "selected_rows": rows.tolist(),
        "selected_episodes": protocol["data"]["selected_episodes"],
        "loss_weights": weights, "arms": arms, "capacity_gates": gates,
        "interpretation": interpret(gates),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"capacity_gates": gates,
                      "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
