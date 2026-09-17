"""Rollout fine-tune the Jenga predictor on its own multi-step predictions.

The shipped checkpoint was trained with num_pred=1, which on the toy system compressed different
outcomes toward the middle attractor and gave 75% basin agreement; rollout fine-tuning after a
teacher-forced start fixed it (HANDOFF findings 1-2). The Jenga checkpoint is already the
teacher-forced stage, so this is the second stage only.

Encoder, proprio encoder, action encoder and decoder stay frozen; only the predictor trains. The
loss is latent MSE between the predicted visual tokens and the encoded truth at every step of the
H=8 + 30-step-hold rollout, so the hold that the monitor depends on is trained explicitly.
Gradient checkpointing keeps the 38-step unroll in memory.
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
                           normalise_actions, normalise_proprio)


def load_items(directory):
    latents, actions, proprio, episodes = [], [], [], []
    for path in sorted(Path(directory).glob("ep*.npz")):
        data = np.load(path, allow_pickle=False)
        latents.append(data["latents"]); actions.append(data["actions"])
        proprio.append(data["proprio"])
        episodes += [str(data["episode_id"])] * len(data["latents"])
    return (np.concatenate(latents), np.concatenate(actions), np.concatenate(proprio),
            np.asarray(episodes))


def assemble(model, visual, proprio, actions, device):
    """Build the predictor's input tokens from PRECOMPUTED visual latents (encoder is frozen)."""
    proprio_emb = model.encode_proprio(normalise_proprio(proprio, device))
    act_emb = model.encode_act(normalise_actions(actions, device))
    patches = visual.shape[2]
    proprio_tiled = proprio_emb.unsqueeze(2).expand(-1, -1, patches, -1)
    act_tiled = act_emb.unsqueeze(2).expand(-1, -1, patches, -1)
    return torch.cat([visual, proprio_tiled.repeat(1, 1, 1, model.num_proprio_repeat),
                      act_tiled.repeat(1, 1, 1, model.num_action_repeat)], dim=3)


def rollout_loss(model, batch_latents, batch_actions, batch_proprio, device, steps,
                 use_checkpoint=True):
    """Latent MSE over an autoregressive rollout; teacher data only warm-starts the history."""
    visual = batch_latents.to(device, torch.float32)
    actions, proprio = batch_actions, batch_proprio  # normalise_* takes CPU tensors
    z = assemble(model, visual[:, :NUM_HIST], proprio[:, :NUM_HIST],
                 actions[:, :NUM_HIST], device)
    visual_dim = visual.shape[-1]
    total, count = 0.0, 0
    for step in range(steps):
        context = z[:, -model.num_hist:]
        predicted = (checkpoint(model.predict, context, use_reentrant=False)
                     if use_checkpoint else model.predict(context))
        new = predicted[:, -1:]
        target = visual[:, NUM_HIST + step: NUM_HIST + step + 1]
        total = total + torch.nn.functional.mse_loss(new[..., :visual_dim], target)
        count += 1
        if step == steps - 1:
            break  # the final predicted frame needs no next action, as in VWorldModel.rollout
        # The next action is known; splice it in exactly as VWorldModel.rollout does.
        next_action = actions[:, NUM_HIST + step: NUM_HIST + step + 1]
        new = model.replace_actions_from_z(new.clone(), normalise_actions(next_action, device))
        z = torch.cat([z, new], dim=1)
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/gtf_data"))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/world_model_gtf.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/gtf_train.json"))
    ap.add_argument("--steps", type=int, default=38, help="rollout length in the loss")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val-episodes", type=int, default=5)
    args = ap.parse_args()

    latents, actions, proprio, episodes = load_items(args.data)
    unique = sorted(set(episodes), key=int)
    val_ids = set(unique[:args.val_episodes])
    is_val = np.isin(episodes, list(val_ids))
    print(f"{len(latents)} items, {len(unique)} episodes, "
          f"{is_val.sum()} validation items from {sorted(val_ids, key=int)}")

    device = "cuda"
    model = load_world_model(args.checkpoint, device)
    for p in model.predictor.parameters():
        p.requires_grad_(True)
    model.predictor.train()
    optimiser = torch.optim.AdamW([p for p in model.predictor.parameters()], lr=args.lr,
                                  weight_decay=0.0)
    tensors = (torch.from_numpy(latents), torch.from_numpy(actions), torch.from_numpy(proprio))
    train_index = np.flatnonzero(~is_val)
    val_index = np.flatnonzero(is_val)

    def evaluate(indices, steps):
        model.predictor.eval()
        losses = []
        with torch.inference_mode():
            for first in range(0, len(indices), args.batch):
                take = indices[first:first + args.batch]
                losses.append(float(rollout_loss(
                    model, tensors[0][take], tensors[1][take], tensors[2][take], device,
                    steps, use_checkpoint=False)))
        model.predictor.train()
        return float(np.mean(losses))

    history = {"validation_rollout_mse": [], "train_rollout_mse": []}
    history["validation_rollout_mse"].append(evaluate(val_index, args.steps))
    print(f"epoch 0 (shipped checkpoint): val rollout MSE {history['validation_rollout_mse'][0]:.5f}",
          flush=True)
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        rng.shuffle(train_index)
        start, running = time.time(), []
        for first in range(0, len(train_index) - args.batch + 1, args.batch):
            take = train_index[first:first + args.batch]
            loss = rollout_loss(model, tensors[0][take], tensors[1][take], tensors[2][take],
                                device, args.steps)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            optimiser.step()
            running.append(float(loss))
            if len(running) % 25 == 0:
                print(f"  epoch {epoch + 1} step {len(running)}/"
                      f"{len(train_index) // args.batch} train {np.mean(running[-25:]):.5f} "
                      f"({time.time() - start:.0f}s)", flush=True)
        history["train_rollout_mse"].append(float(np.mean(running)))
        history["validation_rollout_mse"].append(evaluate(val_index, args.steps))
        print(f"epoch {epoch + 1}: train {history['train_rollout_mse'][-1]:.5f} "
              f"val {history['validation_rollout_mse'][-1]:.5f}", flush=True)
        shipped = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        shipped["predictor"] = model.predictor
        shipped["epoch"] = int(shipped.get("epoch", 0)) + epoch + 1
        torch.save(shipped, args.output)
    report = {"protocol": {
        "stage": "rollout fine-tune only; the shipped checkpoint is the teacher-forced stage",
        "trainable": "predictor only", "loss": "latent MSE at every step of the H+hold rollout",
        "rollout_steps": args.steps, "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
        "training_episodes": "the 43 development-panel episodes; holdout episodes unseen",
        "validation_episodes": sorted(val_ids, key=int)},
        "items": len(latents), "history": history, "output": args.output}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
