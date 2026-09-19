"""W5: train the step-wise graph dynamics model on privileged state.

PLAN_WORLDMODEL W5 (the external plan's Phase 3, oracle-state variant, single-expert stochastic
baseline first). Two stages, in the order the toy system showed is not reversible:

  1. teacher forcing: every recorded transition, Gaussian NLL on the normalised state change plus
     BCE on next-step contacts;
  2. rollout fine-tuning, curriculum 4 -> 12 -> 38 steps: integrate from the true start state with
     the predicted means, penalise the rolled-out state at every step, and at the full horizon add
     branch preservation across the 8 probes of each state (difference matching over ALL pairs,
     divergent and not, so exaggerating every perturbation cannot win).

Tracked, per the plan: safe-pair false separation -- how far apart the model puts probe pairs whose
true endings are practically identical. Grade with eval/jenga_w5_eval.py (W0 response curves, then
the monitor end to end), never with the training loss.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from state_dynamics import (StepGraphMoE, StepGraphNet, contact_targets,  # noqa: E402
                            delta_targets, load_balance, rollout)

HORIZONS = (4, 12, 38)          # in CONTROL steps; scaled by the recording's sub-steps
CONTROL_STEPS = 38
FINAL_POSE = slice(0, 27)


def start_from_full_state(full):
    """70-dim start state (jenga_state_data.full_state) -> the 61-dim per-step layout."""
    return np.concatenate([full[..., 0:27], full[..., 40:58], full[..., 58:70], full[..., 36:40]],
                          axis=-1)


def load(directory):
    """-> trajectories (R, T+1, 61) including the start state, actions (R, T, 4), groups, ids,
    and sub-steps per control action (T = 38 x sub-steps)."""
    trajectories, actions, groups, ids = [], [], [], []
    index, substeps = 0, None
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        start = start_from_full_state(data["start_state"])            # (S, 61)
        traces = data["traces"]                                       # (S, K, T, 61)
        windows = data["actions"]                                     # (S, K, 40, 4)
        n_states, n_probes, steps = traces.shape[:3]
        substeps = steps // CONTROL_STEPS
        first = np.repeat(start[:, None, None], n_probes, axis=1)     # (S, K, 1, 61)
        trajectories.append(np.concatenate([first, traces], axis=2).reshape(-1, steps + 1, 61))
        # Each control target is held for its sub-steps, as the simulator does.
        held = np.repeat(windows[:, :, 2:], substeps, axis=2)          # (S, K, T, 4)
        actions.append(held.reshape(-1, steps, 4))                    # action t drives s_t -> s_t+1
        for s in range(n_states):
            groups.append(np.arange(index, index + n_probes)); index += n_probes
            ids.append(f"{data['episode_id']}:{data['seed']}")
    return (np.concatenate(trajectories).astype(np.float32),
            np.concatenate(actions).astype(np.float32), groups, np.asarray(ids), substeps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/trace_data"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w5_stepnet.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/w5_train.json"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tf-epochs", type=int, default=3)
    ap.add_argument("--tf-batch", type=int, default=2048)
    ap.add_argument("--states-per-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--model", choices=("single", "moe"), default="single")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixes weight initialisation, data order and gate noise; runs before "
                         "2026-09-18 evening had random initialisation and data-order seed 0")
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--balance-weight", type=float, default=0.01,
                    help="collapse regularisation, fixed before training and recorded")
    ap.add_argument("--temperature", type=float, nargs=2, default=(1.0, 0.3),
                    help="gate temperature annealed over teacher forcing, then held")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    trajectories, actions, groups, ids, substeps = load(args.data)
    n_steps = trajectories.shape[1] - 1     # NOT `steps`: the teacher-forcing loop reuses that name
    horizons = tuple(h * substeps for h in HORIZONS)
    unique = sorted(set(ids))
    val_ids = set(unique[:max(1, int(len(unique) * args.val_fraction))])
    val_states = [g for g, i in zip(groups, ids) if i in val_ids]
    train_states = [g for g, i in zip(groups, ids) if i not in val_ids]
    train_rows = np.concatenate(train_states)
    print(f"{len(groups)} states, {len(trajectories)} rollouts, "
          f"{len(train_rows) * n_steps} training transitions ({substeps} sub-steps per action), "
          f"{len(val_states)} validation states",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj = torch.from_numpy(trajectories)
    acts = torch.from_numpy(actions)

    # Normalisation from TRAINING transitions only.
    s_now = traj[train_rows, :-1].reshape(-1, 61)
    s_next = traj[train_rows, 1:].reshape(-1, 61)
    block_delta, grip_delta = delta_targets(s_now, s_next)
    block_scale = block_delta.reshape(-1, 15).std(0).clamp_min(1e-7)
    grip_scale = grip_delta.std(0).clamp_min(1e-7)
    state_scale = s_now.std(0).clamp_min(1e-6)
    delta_scale = (block_scale.to(device), grip_scale.to(device))
    state_scale_d = state_scale.to(device)
    del s_now, s_next, block_delta, grip_delta

    moe = args.model == "moe"
    model = (StepGraphMoE(args.hidden, args.rounds, args.experts) if moe
             else StepGraphNet(args.hidden, args.rounds)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {args.model}: {parameters:,} parameters", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def one_step_loss(state, action, nxt):
        out = model(state, action)
        blocks, grip = delta_targets(state, nxt)
        blocks = blocks / delta_scale[0]; grip = grip / delta_scale[1]
        nll = (0.5 * (out["block_logvar"] + (blocks - out["block_mean"]) ** 2
                      / out["block_logvar"].exp())).mean()
        nll = nll + (0.5 * (out["gripper_logvar"] + (grip - out["gripper_mean"]) ** 2
                            / out["gripper_logvar"].exp())).mean()
        per_block, pairs = contact_targets(nxt)
        bce = F.binary_cross_entropy_with_logits(out["block_contact_logits"], per_block) + \
            F.binary_cross_entropy_with_logits(out["pair_contact_logits"], pairs)
        balance = load_balance(out["gate_probabilities"]) if moe else torch.zeros((), device=device)
        return (nll + bce + args.balance_weight * balance, float(nll.detach()), float(bce.detach()))

    history = {"teacher_forcing": [], "rollout": {}, "validation": {}}

    # ---- Stage 1: teacher forcing over every transition.
    transitions = np.stack(np.meshgrid(train_rows, np.arange(n_steps), indexing="ij"), -1).reshape(-1, 2)
    rng = np.random.default_rng(0 if args.seed is None else args.seed)
    total_tf_steps = args.tf_epochs * int(np.ceil(len(transitions) / args.tf_batch))
    tf_step = 0
    for epoch in range(args.tf_epochs):
        rng.shuffle(transitions)
        start, running = time.time(), []
        for first in range(0, len(transitions), args.tf_batch):
            rows, steps = transitions[first:first + args.tf_batch].T
            state = traj[rows, steps].to(device)
            nxt = traj[rows, steps + 1].to(device)
            action = acts[rows, steps].to(device)
            if moe:   # anneal the gate temperature across teacher forcing
                frac = tf_step / max(total_tf_steps - 1, 1)
                model.temperature = args.temperature[0] + frac * (args.temperature[1]
                                                                  - args.temperature[0])
            tf_step += 1
            loss, nll, bce = one_step_loss(state, action, nxt)
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            running.append((nll, bce))
        mean = np.mean(running, axis=0)
        history["teacher_forcing"].append({"nll": float(mean[0]), "contact_bce": float(mean[1])})
        print(f"teacher forcing epoch {epoch + 1}: nll {mean[0]:.4f} bce {mean[1]:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)

    def rollout_terms(rows, horizon):
        state0 = traj[rows, 0].to(device)
        gates = [] if moe else None
        predicted = rollout(model, state0, acts[rows, :horizon].to(device), delta_scale,
                            collect=gates)
        truth = traj[rows, 1:horizon + 1].to(device)
        err = ((predicted[..., :45] - truth[..., :45]) / state_scale_d[:45]) ** 2
        contact = F.binary_cross_entropy(predicted[..., 45:57].clamp(1e-5, 1 - 1e-5),
                                         truth[..., 45:57])
        terms = {"state": err.mean(), "contact": contact}
        if moe:
            terms["balance"] = load_balance(torch.stack(gates, dim=1))
        return predicted, truth, terms

    def branch_and_safe(predicted, truth, group_sizes):
        """Difference matching over all probe pairs within each state, and safe-pair separation."""
        diff_losses, safe = [], []
        offset = 0
        for size in group_sizes:
            p = predicted[offset:offset + size, -1, FINAL_POSE] / state_scale_d[FINAL_POSE]
            t = truth[offset:offset + size, -1, FINAL_POSE] / state_scale_d[FINAL_POSE]
            i, j = torch.triu_indices(size, size, offset=1, device=p.device)
            diff_losses.append((((p[i] - p[j]) - (t[i] - t[j])) ** 2).mean())
            true_gap = (t[i] - t[j]).norm(dim=1)
            quiet = true_gap < torch.quantile(true_gap, 0.25) + 1e-9
            if quiet.any():
                safe.append(float((p[i] - p[j]).norm(dim=1)[quiet].mean().detach()))
            offset += size
        return torch.stack(diff_losses).mean(), (float(np.mean(safe)) if safe else None)

    def evaluate(horizon):
        model.eval()
        errs, safes, usage = [], [], []
        with torch.no_grad():
            for first in range(0, len(val_states), args.states_per_batch):
                batch = val_states[first:first + args.states_per_batch]
                rows = np.concatenate(batch)
                predicted, truth, terms = rollout_terms(rows, horizon)
                errs.append(float(terms["state"]))
                if moe:
                    gates = []
                    rollout(model, traj[rows, 0].to(device), acts[rows, :horizon].to(device),
                            delta_scale, collect=gates)
                    hard = torch.stack(gates, 1).argmax(-1).reshape(-1)
                    usage.append(torch.bincount(hard, minlength=args.experts).float().cpu())
                if horizon == n_steps:
                    _, safe = branch_and_safe(predicted, truth, [len(g) for g in batch])
                    if safe is not None:
                        safes.append(safe)
        model.train()
        out = {"state_error": float(np.mean(errs)),
               "safe_pair_false_separation": float(np.mean(safes)) if safes else None}
        if moe:
            counts = torch.stack(usage).sum(0)
            out["expert_usage"] = (counts / counts.sum()).tolist()
        return out

    history["validation"]["after_teacher_forcing"] = evaluate(n_steps)
    print("validation after teacher forcing:", history["validation"]["after_teacher_forcing"],
          flush=True)

    # ---- Stage 2: rollout fine-tuning, curriculum over horizons.
    order = np.arange(len(train_states))
    weights = None
    for horizon in horizons:
        rng.shuffle(order)
        start, running = time.time(), []
        for first in range(0, len(order), args.states_per_batch):
            batch = [train_states[i] for i in order[first:first + args.states_per_batch]]
            rows = np.concatenate(batch)
            predicted, truth, terms = rollout_terms(rows, horizon)
            if horizon == horizons[-1]:
                branch, _ = branch_and_safe(predicted, truth, [len(g) for g in batch])
                terms["branch"] = branch
            if weights is None or set(weights) != set(terms):
                # Scale-match to the state term on the first batch of this stage; recorded.
                # The balance term keeps its fixed weight instead: it is a regulariser.
                weights = {k: float(terms["state"].detach()) / max(float(v.detach()), 1e-8)
                           for k, v in terms.items() if k != "balance"}
                if "balance" in terms:
                    weights["balance"] = args.balance_weight
            loss = sum(weights[k] * v for k, v in terms.items())
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            running.append({k: float(v.detach()) for k, v in terms.items()})
        history["rollout"][str(horizon)] = {
            "train": {k: float(np.mean([r[k] for r in running])) for k in running[0]},
            "weights": weights, "seconds": time.time() - start}
        history["validation"][f"after_horizon_{horizon}"] = evaluate(n_steps)
        print(f"rollout horizon {horizon}: train {history['rollout'][str(horizon)]['train']} "
              f"val {history['validation'][f'after_horizon_{horizon}']} "
              f"({time.time() - start:.0f}s)", flush=True)
        weights = None

    torch.save({"model": model.state_dict(), "hidden": args.hidden, "rounds": args.rounds,
                "kind": args.model, "experts": args.experts, "parameters": parameters,
                "substeps": substeps, "seed": args.seed,
                "block_scale": block_scale, "grip_scale": grip_scale,
                "state_scale": state_scale}, args.output)
    Path(args.report).write_text(json.dumps(
        {"protocol": {"plan": "PLAN_WORLDMODEL W5, single-expert stochastic baseline",
                      "architecture": f"graph net, 4 nodes, hidden {args.hidden}, "
                                      f"{args.rounds} message-passing rounds, Gaussian head",
                      "stages": "teacher forcing, then rollout curriculum " + str(HORIZONS),
                      "input": "privileged simulator state (oracle variant)"},
         "model": args.model, "experts": args.experts if moe else 1, "seed": args.seed,
         "parameters": parameters, "balance_weight": args.balance_weight if moe else None,
         "temperature": list(args.temperature) if moe else None,
         "states": len(groups), "history": history, "output": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
