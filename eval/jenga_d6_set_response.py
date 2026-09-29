"""Train and evaluate the frozen D6 direct/set-conditioned response pilot.

The experiment uses TRAIN trajectories only and never reads task or failure labels.  Both arms
predict complete 38-step anonymous pose/velocity responses in one pass.  Their only difference is
whether probe tokens can exchange information through permutation-equivariant self-attention.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
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
from jenga_reobservation_interface import (binary_metrics, candidate_rows, compare_nested,
                                           load_nested, nested_stage_rows, validation_rows)
from neighborhood_topology_loss import neighborhood_terms
from set_response_model import DirectSetResponseModel, reconstruct_states


DATA = ROOT / "results/jenga/d4_neighborhood_data_large.npz"
NESTED = ROOT / "results/jenga/d3_data"
PROTOCOL = ROOT / "results/jenga/d6_set_response_protocol.json"
OUTPUT_DIR = ROOT / "results/jenga/d6_set_response"
REPORT = ROOT / "results/jenga/d6_set_response_pilot.json"
ARMS = ("independent_direct", "set_conditioned")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalizers(start, actions, traces):
    state_mean = start.mean(axis=0)
    state_scale = start.std(axis=0)
    state_floor = max(float(np.sqrt(np.mean(state_scale ** 2))) * 1e-3, 1e-6)
    state_scale = np.maximum(state_scale, state_floor)
    action_mean = actions.mean(axis=(0, 1))
    action_scale = actions.std(axis=(0, 1))
    action_floor = max(float(np.sqrt(np.mean(action_scale ** 2))) * 1e-3, 1e-6)
    action_scale = np.maximum(action_scale, action_floor)
    delta = traces[..., :45] - start[:, None, None, :45]
    response_scale = np.sqrt(np.mean(delta ** 2, axis=(0, 1, 2)))
    response_floor = max(float(np.sqrt(np.mean(response_scale ** 2))) * 1e-3, 1e-6)
    response_scale = np.maximum(response_scale, response_floor)
    return tuple(np.asarray(value, np.float32) for value in (
        state_mean, state_scale, action_mean, action_scale, response_scale))


def tensors(values, device):
    return tuple(torch.as_tensor(value, device=device) for value in values)


def predict(model, start, actions, truth, norms, device):
    state_mean, state_scale, action_mean, action_scale, response_scale = norms
    estimate = model(start, actions, state_mean, state_scale, action_mean, action_scale)
    return reconstruct_states(estimate, start, truth, response_scale)


def loss_terms(model, start, actions, truth, nested_start, nested_actions, nested_truth,
               norms, full_scale, device):
    predicted = predict(model, start, actions, truth, norms, device)
    neighborhood = neighborhood_terms(predicted, truth, actions, full_scale)

    groups = len(nested_start)
    flat_actions = nested_actions.reshape(groups, -1, nested_actions.shape[-2], 4)
    flat_truth = nested_truth.reshape(groups, -1, nested_truth.shape[-2], 61)
    nested_predicted = predict(model, nested_start, flat_actions, flat_truth, norms, device)
    nested_predicted = nested_predicted.reshape_as(nested_truth)
    boundary = boundary_scale_terms(nested_predicted, nested_truth, full_scale)
    return {
        "response": neighborhood.response,
        "topology": neighborhood.topology,
        "commitment": neighborhood.commitment,
        "nested": boundary.total(),
    }


def select_probes(actions, truth, count, rng):
    groups, probes = actions.shape[:2]
    indices = np.stack([rng.permutation(probes)[:count] for _ in range(groups)])
    row = np.arange(groups)[:, None]
    return actions[row, indices], truth[row, indices]


def train_arm(name, initial_state, data, train_rows, nested_train, norms, full_scale,
              weights, args, device):
    model = DirectSetResponseModel(
        hidden=args.hidden, heads=args.heads, layers=args.layers,
        set_conditioned=name == "set_conditioned").to(device)
    model.load_state_dict(initial_state, strict=False)
    # Shared modules have identical initialization. The set-specific context is intentionally the
    # only unmatched block because the independent control contains no attention parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    n_start, n_actions, n_truth = nested_train
    for epoch in range(args.epochs):
        rng = np.random.default_rng([args.seed, epoch])
        order = rng.permutation(train_rows)
        running = []; began = time.time()
        model.train()
        for first in range(0, len(order), args.batch):
            rows = order[first:first + args.batch]
            if len(rows) == 0:
                continue
            probe_count = int(rng.integers(32, data["actions"].shape[1] + 1))
            action_np, truth_np = select_probes(
                data["actions"][rows], data["traces"][rows], probe_count, rng)
            pick = rng.choice(len(n_start), len(rows), replace=len(rows) > len(n_start))
            values = loss_terms(
                model,
                torch.as_tensor(data["start"][rows], device=device),
                torch.as_tensor(action_np, device=device),
                torch.as_tensor(truth_np, device=device),
                torch.as_tensor(n_start[pick], device=device),
                torch.as_tensor(n_actions[pick], device=device),
                torch.as_tensor(n_truth[pick], device=device),
                norms, full_scale, device)
            loss = sum(weights[key] * value for key, value in values.items())
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            running.append({key: float(value.detach()) for key, value in values.items()})
        history.append({
            "epoch": epoch + 1,
            "seconds": time.time() - began,
            "losses": {key: float(np.mean([row[key] for row in running])) for key in weights},
        })
        print(f"{name} epoch {epoch + 1}/{args.epochs}: {history[-1]['losses']}", flush=True)
    return model, history


def batched_predictions(model, start, actions, truth, norms, device, batch=4):
    output = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(start), batch):
            stop = min(first + batch, len(start))
            output.append(predict(
                model, torch.as_tensor(start[first:stop], device=device),
                torch.as_tensor(actions[first:stop], device=device),
                torch.as_tensor(truth[first:stop], device=device), norms, device).cpu().numpy())
    return np.concatenate(output)


def continuous(model, start, actions, truth, norms, full_scale, device):
    predicted = batched_predictions(model, start, actions, truth, norms, device)
    values = neighborhood_terms(
        torch.as_tensor(predicted, device=device), torch.as_tensor(truth, device=device),
        torch.as_tensor(actions, device=device), full_scale)
    return predicted, {"response": float(values.response), "topology": float(values.topology),
                       "commitment": float(values.commitment)}


def invariance_checks(model, start, actions, truth, norms, device):
    model.eval(); count = min(4, len(start)); rng = np.random.default_rng(616)
    base_actions = actions[:count]
    with torch.no_grad():
        base = model(torch.as_tensor(start[:count], device=device),
                     torch.as_tensor(base_actions, device=device), *norms[:4])
        errors = []
        for row in range(count):
            permutation = rng.permutation(actions.shape[1])
            inverse = np.argsort(permutation)
            changed = model(
                torch.as_tensor(start[row:row + 1], device=device),
                torch.as_tensor(base_actions[row:row + 1, permutation], device=device),
                *norms[:4])[0, inverse]
            errors.append(float(torch.max(torch.abs(base[row] - changed))))

        subset = np.sort(rng.choice(actions.shape[1], 32, replace=False))
        reduced = model(torch.as_tensor(start[:count], device=device),
                        torch.as_tensor(base_actions[:, subset], device=device), *norms[:4])
        subset_rms = float(torch.sqrt(torch.mean((base[:, subset] - reduced) ** 2)))
    return {"permutation_max_abs": max(errors), "subset_response_rms": subset_rms}


def evaluate(model, data, val_rows, nested_val, norms, full_scale, device):
    start, actions, truth = (data[name][val_rows] for name in ("start", "actions", "traces"))
    predicted, errors = continuous(model, start, actions, truth, norms, full_scale, device)
    candidate = candidate_rows(actions, truth, predicted)
    n_start, n_actions, n_truth = nested_val
    flat_actions = n_actions.reshape(len(n_start), -1, 38, 4)
    flat_truth = n_truth.reshape(len(n_start), -1, 38, 61)
    nested_predicted = batched_predictions(
        model, n_start, flat_actions, flat_truth, norms, device).reshape(n_truth.shape)
    reference = nested_stage_rows(n_actions, n_truth, full_scale.cpu().numpy())
    nested = compare_nested(reference, nested_stage_rows(
        n_actions, nested_predicted, full_scale.cpu().numpy()))
    checks = invariance_checks(model, start, actions, truth, norms, device)
    return {"candidate": candidate, "continuous": errors, "nested": nested, "checks": checks}


def gate(independent, conditioned):
    comparisons = {
        "candidate_recall_gain": (conditioned["candidate"]["recall"]
                                  - independent["candidate"]["recall"]),
        "candidate_agreement_gain": (conditioned["candidate"]["agreement"]
                                     - independent["candidate"]["agreement"]),
        "candidate_added_positive_change": (
            conditioned["candidate"]["added_positive_rate"]
            - independent["candidate"]["added_positive_rate"]),
        "partition_agreement_gain": (conditioned["candidate"]["mean_partition_agreement"]
                                     - independent["candidate"]["mean_partition_agreement"]),
        "nested_final_recall_gain": (conditioned["nested"]["alarm"]["recall"]
                                     - independent["nested"]["alarm"]["recall"]),
        "nested_final_added_positive_change": (
            conditioned["nested"]["alarm"]["added_positive_rate"]
            - independent["nested"]["alarm"]["added_positive_rate"]),
        "permutation_max_abs": conditioned["checks"]["permutation_max_abs"],
    }
    checks = {
        "candidate_recall": comparisons["candidate_recall_gain"] >= .10,
        "candidate_agreement": comparisons["candidate_agreement_gain"] >= .10,
        "candidate_added_positive": comparisons["candidate_added_positive_change"] <= 0.,
        "partition_agreement": comparisons["partition_agreement_gain"] >= .05,
        "nested_final_recall": comparisons["nested_final_recall_gain"] >= .15,
        "nested_final_added_positive": comparisons["nested_final_added_positive_change"] <= .05,
        "permutation_equivariance": comparisons["permutation_max_abs"] <= 1e-5,
    }
    passed = all(checks.values())
    return {"comparisons": comparisons, "checks": checks, "passes": passed,
            "decision": ("replicate fixed seeds 1-5 on TRAIN" if passed else
                         "stop D6; do not tune on validation or open DEV/TEST")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.report)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}")
    protocol = json.loads(Path(args.protocol).read_text())
    frozen = protocol["training"]
    actual = {"seed": args.seed, "epochs": args.epochs, "learning_rate": args.lr,
              "states_per_batch": args.batch}
    if any(frozen[key] != value for key, value in actual.items()):
        raise SystemExit(f"arguments differ from frozen training protocol: {actual}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    source = np.load(args.data, allow_pickle=False)
    data = {name: source[name] for name in ("start", "actions", "traces", "episode_id")}
    val_rows, heldout = validation_rows(data["episode_id"])
    train_rows = np.setdiff1d(np.arange(len(data["start"])), val_rows)
    n_start, n_actions, n_traces, n_sources = load_nested(args.nested, heldout)
    all_n_start, all_n_actions, all_n_traces, all_n_sources = load_nested(
        args.nested, sorted(set(data["episode_id"].astype(str)), key=int))
    train_mask = np.asarray([source.split("_seed")[0][2:] not in set(heldout)
                             for source in all_n_sources])
    nested_train = (all_n_start[train_mask], all_n_actions[train_mask], all_n_traces[train_mask])
    nested_val = (n_start, n_actions, n_traces)

    normalizer_np = normalizers(
        data["start"][train_rows], data["actions"][train_rows], data["traces"][train_rows])
    norms = tensors(normalizer_np, device)
    full_scale = torch.ones(61, device=device); full_scale[:45] = norms[4]

    torch.manual_seed(args.seed)
    independent_template = DirectSetResponseModel(
        hidden=args.hidden, heads=args.heads, layers=args.layers, set_conditioned=False)
    independent_initial = copy.deepcopy(independent_template.state_dict())
    # Shared encoders and decoder start identically in both arms; attention parameters use the
    # continuation of the same fixed RNG stream.
    conditioned_template = DirectSetResponseModel(
        hidden=args.hidden, heads=args.heads, layers=args.layers, set_conditioned=True)
    conditioned_initial = copy.deepcopy(conditioned_template.state_dict())
    for key in list(conditioned_initial):
        if key in independent_initial and conditioned_initial[key].shape == independent_initial[key].shape:
            conditioned_initial[key] = independent_initial[key].clone()

    # Freeze scale-matched loss weights from the first independent-control batch.
    probe_actions = torch.as_tensor(data["actions"][train_rows[:args.batch]], device=device)
    probe_truth = torch.as_tensor(data["traces"][train_rows[:args.batch]], device=device)
    count = min(args.batch, len(nested_train[0]))
    weighting_model = DirectSetResponseModel(
        hidden=args.hidden, heads=args.heads, layers=args.layers, set_conditioned=False).to(device)
    weighting_model.load_state_dict(independent_initial)
    weighting = loss_terms(
        weighting_model, torch.as_tensor(data["start"][train_rows[:args.batch]], device=device),
        probe_actions, probe_truth,
        torch.as_tensor(nested_train[0][:count], device=device),
        torch.as_tensor(nested_train[1][:count], device=device),
        torch.as_tensor(nested_train[2][:count], device=device), norms, full_scale, device)
    anchor = float(weighting["response"].detach())
    weights = {name: (1. if name == "response" else
                      anchor / max(float(value.detach()), 1e-8))
               for name, value in weighting.items()}
    del weighting_model

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    arms = {}
    for name, initial in zip(ARMS, (independent_initial, conditioned_initial)):
        torch.manual_seed(args.seed)
        model, history = train_arm(name, initial, data, train_rows, nested_train, norms,
                                   full_scale, weights, args, device)
        evaluation = evaluate(model, data, val_rows, nested_val, norms, full_scale, device)
        checkpoint = output_dir / f"{name}_s{args.seed}.pt"
        torch.save({"model": model.state_dict(), "arm": name, "seed": args.seed,
                    "model_config": {"steps": 38, "hidden": args.hidden,
                                     "heads": args.heads, "layers": args.layers,
                                     "set_conditioned": name == "set_conditioned"},
                    "normalizers": [torch.from_numpy(value) for value in normalizer_np],
                    "protocol_sha256": sha256(args.protocol)}, checkpoint)
        arms[name] = {"history": history, "evaluation": evaluation,
                      "checkpoint": str(checkpoint.relative_to(ROOT)),
                      "checkpoint_sha256": sha256(checkpoint)}
        print(json.dumps({name: evaluation}, indent=2), flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    decision = gate(arms["independent_direct"]["evaluation"],
                    arms["set_conditioned"]["evaluation"])
    result = {
        "protocol": protocol,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol),
        "data_sha256": sha256(args.data),
        "heldout_episodes": heldout,
        "train_states": int(len(train_rows)), "validation_states": int(len(val_rows)),
        "nested_train_groups": int(len(nested_train[0])),
        "nested_validation_groups": int(len(nested_val[0])),
        "loss_weights": weights,
        "arms": arms,
        "decision": decision,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": decision}, indent=2))


if __name__ == "__main__":
    main()
