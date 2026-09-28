"""Train one complete-trajectory mixture with universal D4 supervision."""
import argparse
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

from action_branch_monitor import action_branch_partition, smooth_vs_branch_alarm  # noqa: E402
from boundary_scale_loss import boundary_scale_terms  # noqa: E402
from neighborhood_topology_loss import neighborhood_terms  # noqa: E402
from trajectory_mixture import (TrajectoryMixtureGNN, initialise_from_stepnet,  # noqa: E402
                                mixture_terms)
from jenga_d4_train import load_data, partition_agreement  # noqa: E402
from jenga_w6_simple import load_d3  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(model, scales, data, rows, state_scale, device):
    model.eval(); totals, alarm_pairs, partitions, choices = [], [], [], []
    with torch.no_grad():
        for row in rows:
            probes = data["actions"].shape[1]
            start = data["start"][row:row + 1].to(device).repeat_interleave(probes, 0)
            actions = data["actions"][row].to(device)
            predicted, selected = model.predict(start, actions, scales, state_scale)
            truth = data["traces"][row:row + 1].to(device)
            terms = neighborhood_terms(predicted[None], truth, actions[None], state_scale)
            totals.append([float(terms.response), float(terms.topology),
                           float(terms.commitment)])
            choices.extend(selected.cpu().tolist())
            action_np = actions.cpu().numpy()
            truth_np = truth[0, :, :, :45].cpu().numpy()
            pred_np = predicted[:, :, :45].cpu().numpy()
            try:
                truth_alarm = smooth_vs_branch_alarm(action_np, truth_np[:, :18], truth_np).alarm
                pred_alarm = smooth_vs_branch_alarm(action_np, pred_np[:, :18], pred_np).alarm
                _, truth_labels = action_branch_partition(action_np, truth_np[:, :18], truth_np)
                _, pred_labels = action_branch_partition(action_np, pred_np[:, :18], pred_np)
                alarm_pairs.append((truth_alarm, pred_alarm))
                partitions.append(partition_agreement(truth_labels, pred_labels))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass
    means = np.mean(totals, axis=0)
    truth_alarm = np.asarray([pair[0] for pair in alarm_pairs], bool)
    pred_alarm = np.asarray([pair[1] for pair in alarm_pairs], bool)
    occupancy = np.bincount(choices, minlength=model.modes) / max(len(choices), 1)
    return {"states": len(rows), "response": float(means[0]), "topology": float(means[1]),
            "commitment": float(means[2]), "mode_occupancy": occupancy.tolist(),
            "physical_candidates": int(truth_alarm.sum()),
            "predicted_candidates": int(pred_alarm.sum()),
            "alarm_agreement": float(np.mean(truth_alarm == pred_alarm)),
            "candidate_recall": float(pred_alarm[truth_alarm].mean()) if truth_alarm.any() else None,
            "quiet_false_rate": float(pred_alarm[~truth_alarm].mean()) if (~truth_alarm).any() else None,
            "partition_agreement": float(np.mean(partitions))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(
        ROOT / "results/jenga/d4_neighborhood_data_large.npz"))
    parser.add_argument("--init", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--d3-data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--output", default=str(ROOT / "results/jenga/d5_multimodal_s1.pt"))
    parser.add_argument("--report", default=str(ROOT / "results/jenga/d5_multimodal_s1_train.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--states-per-batch", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=.12)
    parser.add_argument("--temperature", type=float, default=.05)
    parser.add_argument("--max-train-states", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data, _, train_rows, val_rows, val_episodes = load_data(args.data, args.val_fraction)
    if args.max_train_states:
        train_rows = train_rows[:args.max_train_states]
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    state_scale = initial["state_scale"].to(args.device)
    scales = (initial["block_scale"].to(args.device), initial["grip_scale"].to(args.device))
    model = initialise_from_stepnet(TrajectoryMixtureGNN(
        initial["hidden"], initial["rounds"], args.modes), initial).to(args.device)
    d3_start, d3_actions, d3_traces, _ = load_d3(args.d3_data, {
        f"{episode}:{seed}" for episode in val_episodes for seed in range(100, 110)})
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed); weights = None; history = []
    for epoch in range(args.epochs):
        model.train(); began = time.time(); running = []
        order = rng.permutation(train_rows)
        for first in range(0, len(order), args.states_per_batch):
            rows = order[first:first + args.states_per_batch]
            groups, probes = len(rows), data["actions"].shape[1]
            start = data["start"][rows].to(args.device)[:, None].expand(
                groups, probes, 61).reshape(groups * probes, 61)
            actions = data["actions"][rows].to(args.device)
            flat_actions = actions.reshape(groups * probes, 38, 4)
            truth = data["traces"][rows].to(args.device)
            mode_trajectories, logits = model.forward_trajectory(
                start, flat_actions, scales, state_scale)
            mixture = mixture_terms(mode_trajectories, logits, truth.reshape(
                groups * probes, 38, 61), state_scale, args.temperature)
            selected = mixture["selected"].reshape(groups, probes, 38, 61)
            relational = neighborhood_terms(selected, truth, actions, state_scale)

            count = min(groups, len(d3_start)); pick = rng.choice(len(d3_start), count, False)
            nested_start = d3_start[pick].to(args.device).repeat_interleave(12, 0)
            nested_actions = d3_actions[pick].reshape(count * 12, 38, 4).to(args.device)
            nested_truth = d3_traces[pick].to(args.device)
            nested_modes, nested_logits = model.forward_trajectory(
                nested_start, nested_actions, scales, state_scale)
            nested_mix = mixture_terms(nested_modes, nested_logits,
                                       nested_truth.reshape(count * 12, 38, 61),
                                       state_scale, args.temperature)
            boundary = boundary_scale_terms(nested_mix["selected"].reshape(
                count, 6, 2, 38, 61), nested_truth, state_scale).total()
            terms = {"response": relational.response, "topology": relational.topology,
                     "commitment": relational.commitment, "boundary": boundary,
                     "nll": mixture["nll"], "balance": mixture["balance"],
                     "routing": mixture["routing"], "entropy": mixture["entropy"]}
            if weights is None:
                anchor = float(terms["response"].detach())
                weights = {name: anchor / max(abs(float(value.detach())), 1e-6)
                           for name, value in terms.items()}
                weights["response"] = 1.0
                weights["routing"] *= .1
                weights["entropy"] *= .01
            loss = sum(weights[name] * value for name, value in terms.items())
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            occupancy = mixture["posterior"].detach().mean(0).cpu().tolist()
            running.append({**{name: float(value.detach()) for name, value in terms.items()},
                            "posterior_occupancy": occupancy})
        validation = (evaluate(model, scales, data, val_rows, state_scale, args.device)
                      if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs else None)
        history.append({"epoch": epoch + 1, "seconds": time.time() - began,
                        "train": {name: float(np.mean([row[name] for row in running]))
                                  for name in terms},
                        "posterior_occupancy": np.mean(
                            [row["posterior_occupancy"] for row in running], axis=0).tolist(),
                        "validation": validation})
        print(f"D5 seed {args.seed} epoch {epoch + 1}/{args.epochs}: "
              f"{validation if validation is not None else 'validation deferred'}", flush=True)

    checkpoint = {"model": model.state_dict(), "kind": "trajectory_mixture",
                  "hidden": initial["hidden"], "rounds": initial["rounds"],
                  "modes": args.modes, "seed": args.seed, "substeps": 1,
                  "orthonormalise": model.orthonormalise,
                  "hard_contacts": model.hard_contacts,
                  "block_scale": initial["block_scale"], "grip_scale": initial["grip_scale"],
                  "state_scale": initial["state_scale"], "init": args.init,
                  "d5_temperature": args.temperature, "d5_weights": weights}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, args.output)
    output_path = Path(args.output).resolve()
    try:
        checkpoint_name = str(output_path.relative_to(ROOT))
    except ValueError:
        checkpoint_name = str(output_path)
    report = {"protocol": "D5 complete-trajectory mixture, universal D4 supervision",
              "seed": args.seed, "modes": args.modes, "epochs": args.epochs,
              "lr": args.lr, "states_per_batch": args.states_per_batch,
              "train_states": len(train_rows), "validation_states": len(val_rows),
              "validation_episodes": val_episodes, "temperature": args.temperature,
              "weights": weights, "history": history, "validation": history[-1]["validation"],
              "checkpoint": checkpoint_name,
              "checkpoint_sha256": sha256(args.output), "test_opened": False}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
