"""Is boundary localisation limited by the representation? Train the same head on TRUE object state.

PLAN_WORLDMODEL, the cheap decider before building an object-centric stack. W2a's discrete head
reproduces jumps (4-5 code switches per response curve) but not their location: the fork/quiet jump
contrast is ~1.0 against reality's 4.14, and doubling the data moved the spread contrast to 1.78
(reality 1.81) while leaving the jump contrast flat. So magnitude was data-limited and sharpness is
not.

This swaps ONLY the input and the prediction target for privileged simulator state: the true poses
of all three blocks at the chunk start, and codes over the true block poses at the ending. Nothing
else changes -- same action features, same codebook procedure, same losses, same response-curve
gate. It is an upper bound, not a deployable monitor: a camera cannot supply this.

  If it localises (fork/quiet jump ratio near reality's 4.1), the representation is the binding
  constraint and an object-centric model (W3) is worth building.
  If it does not, object features will not save it either: the gap is in the action-to-outcome
  mapping, and W3 as planned would be the wrong build.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from jenga_w2_discrete import (EndingHead, HOLD_STEPS, action_features,  # noqa: E402
                               branch_terms, kmeans)


def load_bulk(directory, limit=0, key="start_pose"):
    """Shards -> start state, ending poses, action windows, and state groups.

    key="start_pose" is the blocks-only ablation (bulk shards); key="start_state" is the full
    privileged state (state shards): blocks, blocks relative to the gripper, gripper pose,
    block velocities and contact flags.
    """
    starts, endings, actions, groups, episodes = [], [], [], [], []
    index = 0
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        probes = data["probe_pose"]           # (states, K, holds, 27)
        windows = data["actions"]             # (states, K, 40, 4)
        start_pose = data[key]                # (states, state_dim)
        n_states, n_probes = probes.shape[:2]
        for s in range(n_states):
            groups.append(np.arange(index, index + n_probes))
            index += n_probes
            starts.append(np.repeat(start_pose[s][None], n_probes, axis=0))
            endings.append(probes[s])
            actions.append(windows[s])
            episodes.append(f"{data['episode_id']}:{data['seed']}")
        del data
        if limit and len(groups) >= limit:
            break
    return (np.concatenate(starts).astype(np.float32),
            np.concatenate(endings).astype(np.float32),
            np.concatenate(actions).astype(np.float32),
            groups, np.asarray(episodes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/bulk_data"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w3_privileged_head.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/w3_privileged_train.json"))
    ap.add_argument("--codes", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--branch-weight", type=float, default=1.0)
    ap.add_argument("--limit-states", type=int, default=0)
    ap.add_argument("--state-key", default="start_pose",
                    help="start_pose = blocks only (bulk shards); start_state = full state")
    args = ap.parse_args()

    start_pose, ending_pose, actions, groups, episodes = load_bulk(
        args.data, args.limit_states, args.state_key)
    unique = sorted(set(episodes))
    cut = max(1, int(len(unique) * args.val_fraction))
    val_ids = set(unique[:cut])
    is_val = np.isin(episodes, list(val_ids))
    print(f"{len(groups)} states, {len(start_pose)} rollouts, "
          f"{int(is_val.sum())} validation states", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Poses are millimetres and rotation-matrix columns; standardise so no channel dominates.
    scale = start_pose.std(0).clip(1e-6)
    inputs = torch.from_numpy((start_pose - start_pose.mean(0)) / scale)
    action_input = action_features(torch.from_numpy(actions))
    train_states = [g for i, g in enumerate(groups) if not is_val[i]]
    val_states = [g for i, g in enumerate(groups) if is_val[i]]
    train_rows = np.concatenate(train_states)

    codebooks, targets, truth = {}, {}, {}
    for slot, held in enumerate(HOLD_STEPS):
        values = ending_pose[:, slot]
        centres, _ = kmeans(values[train_rows], args.codes)
        codebooks[held] = centres
        targets[held] = torch.from_numpy(
            ((values[:, None] - centres[None]) ** 2).sum(2).argmin(1)).long()
        truth[held] = torch.from_numpy(values)
        print(f"hold {held}: {args.codes} codes over TRUE block poses", flush=True)

    model = EndingHead(inputs.shape[1], action_input.shape[1], args.codes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    centre_tensors = {held: torch.from_numpy(codebooks[held]).float().to(device)
                      for held in HOLD_STEPS}
    scales = {}

    def run(group, train):
        take = torch.as_tensor(np.asarray(group))
        logits = model(inputs[take].to(device), action_input[take].to(device))
        loss, stats = 0.0, {}
        for slot, held in enumerate(HOLD_STEPS):
            head_logits = logits[slot]
            target = targets[held][take].to(device)
            cross_entropy = nn.functional.cross_entropy(head_logits, target)
            probabilities = head_logits.softmax(1)
            embedding = probabilities @ centre_tensors[held]
            difference, energy = branch_terms(embedding, truth[held][take].to(device))
            scales.setdefault(f"diff_{held}", max(abs(float(difference.detach())), 1e-6))
            scales.setdefault(f"energy_{held}", max(abs(float(energy.detach())), 1e-6))
            scales.setdefault("ce", max(abs(float(cross_entropy.detach())), 1e-6))
            loss = loss + cross_entropy + args.branch_weight * scales["ce"] * (
                difference / scales[f"diff_{held}"] + energy / scales[f"energy_{held}"])
            stats[f"ce_{held}"] = float(cross_entropy.detach())
            stats[f"acc_{held}"] = float((head_logits.argmax(1) == target).float().mean())
        if train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        return stats

    def evaluate():
        model.eval()
        acc = []
        with torch.no_grad():
            for g in val_states:
                acc.append(run(g, train=False))
        model.train()
        return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    history = {"validation": [evaluate()], "train": []}
    print("epoch 0:", history["validation"][0], flush=True)
    rng = np.random.default_rng(0)
    order = np.arange(len(train_states))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        start, running = time.time(), []
        for i in order:
            running.append(run(train_states[i], train=True))
        history["train"].append({k: float(np.mean([r[k] for r in running]))
                                 for k in running[0]})
        history["validation"].append(evaluate())
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"epoch {epoch + 1} ({time.time() - start:.0f}s) "
                  f"train acc30 {history['train'][-1]['acc_30']:.3f} "
                  f"val acc30 {history['validation'][-1]['acc_30']:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "codebooks": codebooks,
                    "input_mean": start_pose.mean(0), "input_scale": scale,
                    "sizes": {held: {"codes": args.codes} for held in HOLD_STEPS},
                    "input_dim": inputs.shape[1], "action_dim": action_input.shape[1]},
                   args.output)
    Path(args.report).write_text(json.dumps(
        {"protocol": {"plan": "PLAN_WORLDMODEL W3 decider (privileged upper bound)",
                      "input": args.state_key,
                      "target": "codes over TRUE block poses at each hold step",
                      "codes": args.codes, "epochs": args.epochs,
                      "not_deployable": "a camera cannot supply this input"},
         "states": len(groups), "history": history, "output": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
