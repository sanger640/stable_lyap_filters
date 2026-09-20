"""W6: the simple world model. One-step training with input noise, no curriculum.

Everything the audit could not justify is gone:

  * control-rate data (38 steps per rollout), not the 5x finer recording. The finer data was added
    because a topple's onset happens inside one 0.1 s control step, but MoE A predicts a topple at
    0% of fork states and still scores 46% on Gate 3, so topple prediction is not what the monitor
    reads. Same 8,810 states and 70,480 rollouts either way -- only the resolution differs.
  * no rollout curriculum and no backprop through 190 steps. Compounding error is handled the way
    the published models in this family do it (GNS, MeshGraphNets): corrupt the input state with
    noise during one-step training so the model sees states like the ones its own rollout visits,
    and learn to correct back towards the truth.
  * no branch-preservation term, no transition weighting, no mixture of experts. Those are the
    additions to test LATER, one at a time, against this baseline.

The architecture is unchanged (`src/state_dynamics.StepGraphNet`) and the checkpoint is the same
format, so `eval/jenga_w5_eval.py` and `eval/jenga_w5_gate3.py` grade this with no changes and the
numbers are directly comparable to every W5 model.
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
from state_dynamics import (StepGraphNet, contact_targets,  # noqa: E402
                            delta_targets, neighbour_tilt_deg, rollout)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_w5_train import load  # noqa: E402

CONTINUOUS = list(range(0, 45)) + [57, 58, 59]   # pose, rotation, velocity, gripper position
TOPPLE_DEG = 45.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/trace_data"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w6_simple.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/w6_simple_train.json"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.5,
                    help="input-state noise, in units of the per-dimension standard deviation of "
                         "one true step; 0 disables it (GNS-style corruption)")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    trajectories, actions, groups, ids, substeps = load(args.data)
    n_steps = trajectories.shape[1] - 1
    unique = sorted(set(ids))
    val_ids = set(unique[:max(1, int(len(unique) * args.val_fraction))])
    val_states = [g for g, i in zip(groups, ids) if i in val_ids]
    train_states = [g for g, i in zip(groups, ids) if i not in val_ids]
    train_rows = np.concatenate(train_states)
    val_rows = np.concatenate(val_states)
    print(f"{len(groups)} states, {len(trajectories)} rollouts, {n_steps} steps each "
          f"({substeps} sub-step(s) per action), {len(train_rows) * n_steps} training transitions",
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
    # One true step's size per dimension: the scale the input corruption is measured in.
    step_sigma = (s_next - s_now).std(0).clamp_min(1e-9).to(device)
    delta_scale = (block_scale.to(device), grip_scale.to(device))
    state_scale_d = state_scale.to(device)
    del s_now, s_next, block_delta, grip_delta

    model = StepGraphNet(args.hidden, args.rounds).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model single: {parameters:,} parameters", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def transitions_of(rows):
        pairs = np.stack(np.meshgrid(rows, np.arange(n_steps), indexing="ij"), -1)
        return pairs.reshape(-1, 2)

    train_transitions = transitions_of(train_rows)
    rng = np.random.default_rng(0 if args.seed is None else args.seed)

    def corrupt(state):
        """GNS-style input noise: the model must map a drifted state back onto the true next one."""
        if args.noise <= 0:
            return state
        noisy = state.clone()
        index = torch.tensor(CONTINUOUS, device=state.device)
        noisy[:, index] = state[:, index] + args.noise * step_sigma[index] * torch.randn(
            len(state), len(CONTINUOUS), device=state.device)
        return noisy

    def one_step_loss(state, action, nxt):
        out = model(state, action)
        # Targets are measured from the CORRUPTED state, so the prediction lands on the true next.
        blocks, grip = delta_targets(state, nxt)
        blocks = blocks / delta_scale[0]
        grip = grip / delta_scale[1]
        mse = F.mse_loss(out["block_mean"], blocks) + F.mse_loss(out["gripper_mean"], grip)
        per_block, pairs = contact_targets(nxt)
        bce = F.binary_cross_entropy_with_logits(out["block_contact_logits"], per_block) + \
            F.binary_cross_entropy_with_logits(out["pair_contact_logits"], pairs)
        return mse + bce, mse, bce

    def evaluate():
        """One-step error, and the full-horizon rollout the monitor actually uses."""
        model.eval()
        step_err, roll_err, predicted_flags, true_flags = [], [], [], []
        with torch.no_grad():
            for first in range(0, len(val_rows), 256):
                rows = val_rows[first:first + 256]
                state = traj[rows, :-1].reshape(-1, 61).to(device)
                nxt = traj[rows, 1:].reshape(-1, 61).to(device)
                action = acts[rows].reshape(-1, 4).to(device)
                _, mse, _ = one_step_loss(state, action, nxt)
                step_err.append(float(mse))
                predicted = rollout(model, traj[rows, 0].to(device), acts[rows].to(device),
                                    delta_scale)
                truth = traj[rows, 1:].to(device)
                roll_err.append(float((((predicted[..., :45] - truth[..., :45])
                                        / state_scale_d[:45]) ** 2).mean()))
                eligible = neighbour_tilt_deg(traj[rows, 0].numpy()) < TOPPLE_DEG
                predicted_flags.append((neighbour_tilt_deg(predicted[:, -1].cpu().numpy())
                                        >= TOPPLE_DEG) & eligible)
                true_flags.append((neighbour_tilt_deg(truth[:, -1].cpu().numpy())
                                   >= TOPPLE_DEG) & eligible)
        model.train()
        predicted_flags = np.concatenate(predicted_flags).any(-1)
        true_flags = np.concatenate(true_flags).any(-1)
        return {"one_step_mse": float(np.mean(step_err)),
                "rollout_state_error": float(np.mean(roll_err)),
                "topple_recall": (float(predicted_flags[true_flags].mean())
                                  if true_flags.any() else None),
                "topple_false_rate": (float(predicted_flags[~true_flags].mean())
                                      if (~true_flags).any() else None)}

    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train_transitions)
        start, running = time.time(), []
        for first in range(0, len(train_transitions), args.batch):
            rows, steps = train_transitions[first:first + args.batch].T
            state = corrupt(traj[rows, steps].to(device))
            nxt = traj[rows, steps + 1].to(device)
            loss, mse, bce = one_step_loss(state, acts[rows, steps].to(device), nxt)
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            running.append((float(mse.detach()), float(bce.detach())))
        validation = evaluate()
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(running, axis=0)[0]),
                        "train_bce": float(np.mean(running, axis=0)[1]),
                        "validation": validation, "seconds": time.time() - start})
        print(f"epoch {epoch + 1}/{args.epochs}: train mse {history[-1]['train_mse']:.4f} "
              f"val {validation} ({time.time() - start:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "hidden": args.hidden, "rounds": args.rounds,
                "kind": "single", "experts": 1, "parameters": parameters,
                "substeps": substeps, "seed": args.seed, "noise": args.noise,
                "transition_weighting": False,
                "block_scale": block_scale, "grip_scale": grip_scale,
                "state_scale": state_scale}, args.output)
    Path(args.report).write_text(json.dumps(
        {"protocol": {"plan": "W6 simple baseline: one-step training with input noise",
                      "architecture": f"graph net, 4 nodes, hidden {args.hidden}, "
                                      f"{args.rounds} rounds",
                      "training": f"{args.epochs} epochs, one-step MSE + contact BCE, input noise "
                                  f"{args.noise} x one-step sigma, no curriculum, no branch loss",
                      "input": "privileged simulator state (oracle variant)"},
         "seed": args.seed, "noise": args.noise, "parameters": parameters,
         "states": len(groups), "history": history, "output": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
