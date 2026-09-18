"""W1: train the predictor on SETS of perturbed rollouts from the same state.

PLAN_WORLDMODEL W1. The 2026-09-17 fine-tune cut rollout MSE 10x and collapsed the model
(predicted spread 0.18x reality, jump ratio 2.3 against reality's 14.9), because
  * each state was seen with exactly one perturbation, so "the action caused this" and "this state
    usually looks like that" are not distinguishable, and
  * 30 of the 38 steps are a held pose where almost nothing moves, so 79% of the loss terms teach
    stasis, and MSE at a fork is minimised by the blend.

One item here is one simulator state with K perturbed rollouts, and the loss sees them together:

  paired      mean_k ||z^_k - z_k||^2                        -- stay anchored to reality
  energy      (1/K^2) sum_ij ||z^_i - z_j||                  -- proper scoring rule; the negative
              - (1/2K^2) sum_ij ||z^_i - z^_j||                 term rewards spread, so a collapsed
                                                                 set is penalised (norms, NOT squares)
  difference  mean_ij ||(z^_i - z^_j) - (z_i - z_j)||^2      -- the spread must be action-caused;
                                                                 this is the geometry the monitor reads
  temporal    ||(z^_t - z^_{t-1}) - (z_t - z_{t-1})||^2      -- "nothing happens" stops being free

Steps are also weighted by how much the true scene actually changes, so the held tail cannot
dominate. Term weights are scale-matched on the first batch rather than tuned, and the choice is
recorded in the report. Grade with eval/jenga_w0_response_curves.py, never with rollout MSE.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, NUM_HIST, load_world_model,  # noqa: E402
                           normalise_actions)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_gtf_train import assemble  # noqa: E402


def load_groups(directory):
    """Returns latents, actions, proprio, and the index groups that share a simulator state."""
    latents, actions, proprio, states, episodes = [], [], [], [], []
    for path in sorted(Path(directory).glob("ep*.npz")):
        data = np.load(path, allow_pickle=False)
        latents.append(data["latents"]); actions.append(data["actions"])
        proprio.append(data["proprio"]); states.append(data["state_ids"])
        episodes += [str(data["episode_id"])] * len(data["latents"])
    latents = np.concatenate(latents); states = np.concatenate(states)
    groups = {}
    for i, s in enumerate(states):
        groups.setdefault(str(s), []).append(i)
    return (latents, np.concatenate(actions), np.concatenate(proprio),
            np.asarray(episodes), [np.asarray(v) for v in groups.values()])


def predict_rollout(model, latents, actions, proprio, device, steps, use_checkpoint=True):
    """Autoregressive rollout; returns predicted visual latents (b, steps, patches, dim)."""
    visual = latents.to(device, torch.float32)
    z = assemble(model, visual[:, :NUM_HIST], proprio[:, :NUM_HIST], actions[:, :NUM_HIST], device)
    dim = visual.shape[-1]
    out = []
    for step in range(steps):
        context = z[:, -model.num_hist:]
        predicted = (checkpoint(model.predict, context, use_reentrant=False)
                     if use_checkpoint else model.predict(context))
        new = predicted[:, -1:]
        out.append(new[..., :dim])
        if step == steps - 1:
            break
        next_action = actions[:, NUM_HIST + step: NUM_HIST + step + 1]
        new = model.replace_actions_from_z(new.clone(), normalise_actions(next_action, device))
        z = torch.cat([z, new], dim=1)
    return torch.cat(out, dim=1), visual[:, NUM_HIST:NUM_HIST + steps]


def step_weights(truth):
    """Weight each step by how much the true scene moves, mean 1, so the hold cannot dominate."""
    change = (truth[:, 1:] - truth[:, :-1]).flatten(2).norm(dim=2).mean(0)
    change = torch.cat([change[:1], change])
    return change / change.mean().clamp_min(1e-8)


def set_terms(predicted, truth, weights):
    """predicted/truth: (K, steps, patches, dim) for ONE state. Returns the four loss terms."""
    k = predicted.shape[0]
    w = weights.view(1, -1, 1, 1)
    paired = (((predicted - truth) ** 2) * w).mean()
    # Endings only for the set terms: that is what the monitor reads.
    p_end, t_end = predicted[:, -1].flatten(1), truth[:, -1].flatten(1)
    cross = torch.cdist(p_end, t_end).mean()
    within = torch.cdist(p_end, p_end).sum() / (k * k)
    energy = cross - 0.5 * within
    i, j = torch.triu_indices(k, k, offset=1, device=p_end.device)
    difference = (((p_end[i] - p_end[j]) - (t_end[i] - t_end[j])) ** 2).mean()
    dp = predicted[:, 1:] - predicted[:, :-1]
    dt = truth[:, 1:] - truth[:, :-1]
    temporal = (((dp - dt) ** 2) * w[:, 1:]).mean()
    return {"paired": paired, "energy": energy, "difference": difference, "temporal": temporal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/w1_data"))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="start from the SHIPPED checkpoint, not the collapsed fine-tune")
    ap.add_argument("--output", default=str(ROOT / "results/jenga/world_model_w1.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/w1_train.json"))
    ap.add_argument("--steps", type=int, default=38)
    ap.add_argument("--states-per-batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val-episodes", type=int, default=5)
    args = ap.parse_args()

    latents, actions, proprio, episodes, groups = load_groups(args.data)
    unique = sorted(set(episodes), key=int)
    val_ids = set(unique[:args.val_episodes])
    train_groups = [g for g in groups if episodes[g[0]] not in val_ids]
    val_groups = [g for g in groups if episodes[g[0]] in val_ids]
    print(f"{len(latents)} rollouts, {len(groups)} states, K={len(groups[0])}, "
          f"{len(val_groups)} validation states", flush=True)

    device = "cuda"
    model = load_world_model(args.checkpoint, device)
    for p in model.predictor.parameters():
        p.requires_grad_(True)
    model.predictor.train()
    optimiser = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr, weight_decay=0.0)
    tensors = (torch.from_numpy(latents), torch.from_numpy(actions), torch.from_numpy(proprio))

    def batch_terms(group, use_checkpoint=True):
        take = torch.as_tensor(np.asarray(group))
        predicted, truth = predict_rollout(model, tensors[0][take], tensors[1][take],
                                           tensors[2][take], device, args.steps, use_checkpoint)
        return set_terms(predicted, truth, step_weights(truth))

    # Scale-match the weights on the first training state: no tuning, recorded in the report.
    with torch.no_grad():
        first = {k: float(v) for k, v in batch_terms(train_groups[0], False).items()}
    weights = {k: (abs(first["paired"]) / max(abs(v), 1e-8)) for k, v in first.items()}
    print("scale-matched weights:", {k: round(v, 5) for k, v in weights.items()}, flush=True)

    def total(terms):
        return sum(weights[k] * v for k, v in terms.items())

    def evaluate(group_list):
        model.predictor.eval()
        acc = []
        with torch.inference_mode():
            for g in group_list:
                terms = batch_terms(g, use_checkpoint=False)
                acc.append([float(terms[k]) for k in ("paired", "energy", "difference")])
        model.predictor.train()
        mean = np.mean(acc, axis=0)
        return {"paired": float(mean[0]), "energy": float(mean[1]),
                "difference": float(mean[2])}

    history = {"validation": [evaluate(val_groups)], "train": []}
    print("epoch 0 (start):", history["validation"][0], flush=True)
    order = np.arange(len(train_groups))
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        rng.shuffle(order)
        start, running = time.time(), []
        for count, index in enumerate(order, 1):
            terms = batch_terms(train_groups[index])
            loss = total(terms)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            optimiser.step()
            running.append([float(terms[k]) for k in ("paired", "energy", "difference")])
            if count % 25 == 0:
                recent = np.mean(running[-25:], axis=0)
                print(f"  epoch {epoch + 1} state {count}/{len(order)} "
                      f"paired {recent[0]:.4f} energy {recent[1]:.2f} diff {recent[2]:.2f} "
                      f"({time.time() - start:.0f}s)", flush=True)
        mean = np.mean(running, axis=0)
        history["train"].append({"paired": float(mean[0]), "energy": float(mean[1]),
                                 "difference": float(mean[2])})
        history["validation"].append(evaluate(val_groups))
        print(f"epoch {epoch + 1}: train {history['train'][-1]} "
              f"val {history['validation'][-1]}", flush=True)
        shipped = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        shipped["predictor"] = model.predictor
        shipped["epoch"] = int(shipped.get("epoch", 0)) + epoch + 1
        torch.save(shipped, args.output)
    report = {"protocol": {
        "plan": "PLAN_WORLDMODEL W1", "start_from": args.checkpoint,
        "loss": "paired MSE + energy score + difference matching + temporal difference, "
                "steps weighted by true scene change",
        "weights_scale_matched_on_first_state": weights,
        "K_per_state": int(len(groups[0])), "states_per_batch": args.states_per_batch,
        "rollout_steps": args.steps, "epochs": args.epochs, "lr": args.lr,
        "grade_with": "eval/jenga_w0_response_curves.py (spread and jump ratio), never MSE"},
        "states": len(groups), "history": history, "output": args.output}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(history["validation"], indent=1))


if __name__ == "__main__":
    main()
