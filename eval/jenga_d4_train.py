"""One-seed matched D4 architecture gate on TRAIN-only counterfactual neighbourhoods."""
import argparse
import copy
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
from state_dynamics import rollout  # noqa: E402
from jenga_w5_eval import load_model  # noqa: E402
from jenga_w6_simple import load_d3  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_data(path, val_fraction):
    data = np.load(path, allow_pickle=False)
    episodes = data["episode_id"].astype(str)
    unique = sorted(set(episodes), key=int)
    heldout = set(unique[:max(1, int(np.ceil(len(unique) * val_fraction)))])
    val = np.asarray([episode in heldout for episode in episodes])
    return {name: torch.from_numpy(data[name]) for name in ("start", "actions", "traces")}, \
        episodes, np.flatnonzero(~val), np.flatnonzero(val), sorted(heldout, key=int)


def partition_agreement(left, right):
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    return float(max(np.mean(left == right), np.mean(left != right)))


def evaluate(model, scales, data, rows, state_scale, device):
    model.eval(); totals = []; alarm_pairs = []; partitions = []
    with torch.no_grad():
        for row in rows:
            start = data["start"][row:row + 1].to(device).repeat_interleave(
                data["actions"].shape[1], 0)
            actions = data["actions"][row].to(device)
            predicted = rollout(model, start, actions, scales).unsqueeze(0)
            truth = data["traces"][row:row + 1].to(device)
            terms = neighborhood_terms(predicted, truth, actions.unsqueeze(0), state_scale)
            totals.append([float(terms.response), float(terms.topology),
                           float(terms.commitment)])
            action_np = actions.cpu().numpy()
            truth_np = truth[0, :, :, :45].cpu().numpy()
            pred_np = predicted[0, :, :, :45].cpu().numpy()
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
    return {"states": len(rows), "response": float(means[0]), "topology": float(means[1]),
            "commitment": float(means[2]),
            "physical_candidates": int(truth_alarm.sum()),
            "predicted_candidates": int(pred_alarm.sum()),
            "alarm_agreement": float(np.mean(truth_alarm == pred_alarm)) if len(alarm_pairs) else None,
            "candidate_recall": (float(pred_alarm[truth_alarm].mean()) if truth_alarm.any() else None),
            "quiet_false_rate": (float(pred_alarm[~truth_alarm].mean())
                                 if (~truth_alarm).any() else None),
            "partition_agreement": float(np.mean(partitions)) if partitions else None}


def save_checkpoint(path, template, model, arm, protocol):
    state = copy.deepcopy(template); state["model"] = model.state_dict()
    state.update({"seed": protocol["seed"], "d4_arm": arm, "d4_protocol": protocol})
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "results/jenga/d4_neighborhood_data.npz"))
    parser.add_argument("--init", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--d3-data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/jenga/d4"))
    parser.add_argument("--report", default=str(ROOT / "results/jenga/d4_one_seed_summary.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=.12)
    parser.add_argument("--states-per-batch", type=int, default=1)
    parser.add_argument("--arms", default="matched_continuation,topology,full_d4",
                        help="comma-separated continuation arms; frozen D2 is always evaluated")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="evaluate every N epochs and always on the final epoch")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-states", type=int, default=None)
    args = parser.parse_args()
    requested_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    valid_arms = {"matched_continuation", "topology", "full_d4"}
    if not requested_arms or not set(requested_arms) <= valid_arms:
        parser.error(f"--arms must contain only {sorted(valid_arms)}")
    if args.states_per_batch < 1 or args.eval_every < 1:
        parser.error("--states-per-batch and --eval-every must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    data, episodes, train_rows, val_rows, val_episodes = load_data(args.data, args.val_fraction)
    if args.max_train_states:
        train_rows = train_rows[:args.max_train_states]
    if not len(train_rows) or not len(val_rows):
        raise SystemExit("D4 requires at least one training and one validation state")
    template = torch.load(args.init, map_location="cpu", weights_only=False)
    state_scale = template["state_scale"].to(device)
    d3_start, d3_actions, d3_traces, _ = load_d3(args.d3_data, {
        f"{episode}:{seed}" for episode in val_episodes for seed in range(100, 110)})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    protocol = {"name": "D4 one-seed full-neighbourhood architecture gate",
                "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
                "train_states": len(train_rows), "validation_states": len(val_rows),
                "validation_episodes": val_episodes, "test_opened": False,
                "states_per_batch": args.states_per_batch,
                "arms": ["frozen_d2", *requested_arms],
                "full_d4": "response + complete topology + local commitment + D3 boundary",
                "selection": "unlabeled episode split and final validation objective"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    frozen, frozen_scales = load_model(args.init, device)
    arms = {"frozen_d2": {"validation": evaluate(
        frozen, frozen_scales, data, val_rows, state_scale, device), "history": []}}
    rng = np.random.default_rng(args.seed)
    for arm in requested_arms:
        model, scales = load_model(args.init, device); model.train()
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
        weights = None; history = []
        for epoch in range(args.epochs):
            order = rng.permutation(train_rows); running = []; began = time.time()
            for first in range(0, len(order), args.states_per_batch):
                rows = order[first:first + args.states_per_batch]
                groups = len(rows)
                probes = data["actions"].shape[1]
                start = data["start"][rows].to(device)[:, None].expand(
                    groups, probes, 61).reshape(groups * probes, 61)
                actions = data["actions"][rows].to(device)
                truth = data["traces"][rows].to(device)
                predicted = rollout(model, start, actions.reshape(
                    groups * probes, actions.shape[2], 4), scales).reshape_as(truth)
                terms = neighborhood_terms(predicted, truth, actions, state_scale)
                selected = {"response": terms.response}
                if arm in ("topology", "full_d4"):
                    selected["topology"] = terms.topology
                if arm == "full_d4":
                    selected["commitment"] = terms.commitment
                    count = min(groups, len(d3_start))
                    pick = rng.choice(len(d3_start), count, replace=False)
                    d3_pred = rollout(model, d3_start[pick].to(device).repeat_interleave(
                        12, 0), d3_actions[pick].reshape(count * 12, 38, 4).to(device), scales)
                    boundary = boundary_scale_terms(
                        d3_pred.reshape(count, 6, 2, 38, 61),
                        d3_traces[pick].to(device), state_scale)
                    selected["boundary"] = boundary.total()
                if weights is None:
                    anchor = float(selected["response"].detach())
                    weights = {name: (1.0 if name == "response" else
                                      anchor / max(float(value.detach()), 1e-8))
                               for name, value in selected.items()}
                loss = sum(weights[name] * value for name, value in selected.items())
                optimiser.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
                running.append({name: float(value.detach()) for name, value in selected.items()})
            should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
            validation = (evaluate(model, scales, data, val_rows, state_scale, device)
                          if should_evaluate else None)
            history.append({"epoch": epoch + 1, "seconds": time.time() - began,
                            "train": {name: float(np.mean([row[name] for row in running]))
                                      for name in running[0]}, "validation": validation})
            print(f"{arm} epoch {epoch + 1}/{args.epochs}: "
                  f"{validation if validation is not None else 'validation deferred'}", flush=True)
        checkpoint = output / f"{arm}_s{args.seed}.pt"
        save_checkpoint(checkpoint, template, model, arm, protocol)
        arms[arm] = {"weights": weights, "history": history,
                     "validation": history[-1]["validation"],
                     "checkpoint": display_path(checkpoint),
                     "checkpoint_sha256": sha256(checkpoint)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {"protocol": protocol, "data_sha256": sha256(args.data),
              "initial_checkpoint": display_path(args.init),
              "initial_checkpoint_sha256": sha256(args.init), "arms": arms,
              "decision": "pending"}
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
