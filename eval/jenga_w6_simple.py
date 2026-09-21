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
from state_dynamics import (StepGraphMoE, StepGraphNet, StepMLP, contact_targets,  # noqa: E402
                            delta_targets, load_balance, neighbour_tilt_deg, rollout)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_w5_train import load  # noqa: E402

CONTINUOUS = list(range(0, 45)) + [57, 58, 59]   # pose, rotation, velocity, gripper position
TOPPLE_DEG = 45.0
POSE = slice(0, 27)
SEPARATION = 1.0    # a pair counts as truly separating at one std of the normalised pose


def branch_terms(predicted, truth, group_sizes, state_scale):
    """Branch separation over the WHOLE rollout, weighted towards persistent divergence.

    For every probe pair in a state, the predicted distance between the two trajectories must
    match the true distance at every step, with a linear ramp so late (persistent) divergence
    counts more than an early transient. Matching the distance rather than the difference VECTOR
    is deliberate: the monitor scores the spread of endings, so magnitude is what must be right,
    and demanding the direction too is a stronger requirement than the task needs.

    Returns (loss, mean predicted/true end distance ratio over pairs that truly separate).
    """
    steps = predicted.shape[1]
    ramp = torch.linspace(0.0, 1.0, steps, device=predicted.device)
    losses, ratios = [], []
    offset = 0
    for size in group_sizes:
        p = predicted[offset:offset + size, :, POSE] / state_scale[POSE]
        q = truth[offset:offset + size, :, POSE] / state_scale[POSE]
        i, j = torch.triu_indices(size, size, offset=1, device=p.device)
        predicted_gap = (p[i] - p[j]).norm(dim=-1)                      # (pairs, steps)
        true_gap = (q[i] - q[j]).norm(dim=-1)
        losses.append((ramp * (predicted_gap - true_gap) ** 2).mean())
        # Ratio over pairs that TRULY separate, aggregated rather than averaged per pair: a
        # per-pair ratio divides by near-zero true gaps at quiet states and explodes.
        separating = true_gap[:, -1] > SEPARATION
        if separating.any():
            ratios.append((float(predicted_gap[separating, -1].sum().detach()),
                           float(true_gap[separating, -1].sum().detach())))
        offset += size
    if not ratios:
        return torch.stack(losses).mean(), None
    predicted_total = sum(r[0] for r in ratios)
    return torch.stack(losses).mean(), predicted_total / max(sum(r[1] for r in ratios), 1e-9)


def branch_terms_batched(predicted, truth, branches, state_scale):
    """`branch_terms` for groups of equal size, in one batched pass instead of a Python loop.

    predicted, truth: (G * branches, steps, 61), groups contiguous. Same loss, same ratio: the
    per-group mean of ramp * (predicted gap - true gap)^2, then the mean over groups.
    """
    steps = predicted.shape[1]
    groups = predicted.shape[0] // branches
    ramp = torch.linspace(0.0, 1.0, steps, device=predicted.device)
    p = (predicted[:, :, POSE] / state_scale[POSE]).reshape(groups, branches, steps, -1)
    q = (truth[:, :, POSE] / state_scale[POSE]).reshape(groups, branches, steps, -1)
    i, j = torch.triu_indices(branches, branches, offset=1, device=p.device)
    predicted_gap = (p[:, i] - p[:, j]).norm(dim=-1)                   # (G, pairs, steps)
    true_gap = (q[:, i] - q[:, j]).norm(dim=-1)
    loss = (ramp * (predicted_gap - true_gap) ** 2).mean(dim=(1, 2)).mean()
    separating = true_gap[:, :, -1] > SEPARATION
    if not separating.any():
        return loss, None
    ratio = float(predicted_gap[:, :, -1][separating].sum().detach()) / max(
        float(true_gap[:, :, -1][separating].sum().detach()), 1e-9)
    return loss, ratio


def intervention_terms(predicted, truth, cont_scale, diff_scale, quiet_tau):
    """CoCo-inspired intervention consistency over a short unroll (PLAN_NEXT.md Phase 3, D2).

    predicted, truth: (G, B, H, 61) -- G branch points, B branches from the SAME true state
    (branch 0 the nominal action, the rest nearby actions), H control steps. Only the generic
    continuous state enters (object pose, 6D orientation, linear and angular velocity, gripper
    position); nothing in here knows about blocks standing, tipping or falling.

      response    each branch's predicted state matches its real one at every step -- the absolute
                  response to that action, so a bias shared by all branches is also penalised
      difference  the INTERVENTION EFFECT  Delta(t) = x^{a+d}_t - x^a_t  matches in magnitude,
                  direction and timing (onset and persistence), at every step
      quiet       extra weight on the effect mismatch where the real intervention changes (almost)
                  nothing by the end of the unroll, so benign interventions cannot create
                  artificial divergence. It penalises (Delta_pred - Delta_real)^2, NOT Delta_pred^2:
                  "small" is not zero, and pushing small real effects to zero would bias the model
                  towards under-response -- the very failure this objective exists to fix.

    All terms are normalised: states by the per-dimension state scale, effects by the per-dimension
    scale of real intervention effects in the training data, so each is O(1).
    """
    x_hat = predicted[..., CONTINUOUS] / cont_scale
    x = truth[..., CONTINUOUS] / cont_scale
    response = ((x_hat - x) ** 2).mean()
    d_hat = (predicted[:, 1:, :, CONTINUOUS] - predicted[:, :1, :, CONTINUOUS]) / diff_scale
    d = (truth[:, 1:, :, CONTINUOUS] - truth[:, :1, :, CONTINUOUS]) / diff_scale
    difference = ((d_hat - d) ** 2).mean()
    quiet_pairs = d[:, :, -1].norm(dim=-1) < quiet_tau                       # (G, B-1)
    mismatch = (d_hat - d) ** 2
    quiet = mismatch[quiet_pairs].mean() if quiet_pairs.any() else mismatch.sum() * 0.0
    return {"response": response, "difference": difference, "quiet": quiet}


def load_cw(directory, exclude_ids):
    """Contact-window branch groups (`jenga_cw_data.py`), minus any from held-out training files."""
    starts, actions, traces, sources = [], [], [], []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        ident = path.stem.replace("ep", "").replace("_seed", ":")
        if ident in exclude_ids:
            continue
        d = np.load(path, allow_pickle=False)
        if not len(d["start"]):
            continue
        starts.append(d["start"]); actions.append(d["actions"]); traces.append(d["traces"])
        sources.extend([ident] * len(d["start"]))
    if not starts:
        raise SystemExit(f"no contact-window groups left in {directory} after excluding the "
                         f"validation episodes")
    return (torch.from_numpy(np.concatenate(starts)), torch.from_numpy(np.concatenate(actions)),
            torch.from_numpy(np.concatenate(traces)), sources)


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
    ap.add_argument("--model", choices=("single", "moe", "mlp"), default="single")
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--balance-weight", type=float, default=1.0)
    ap.add_argument("--rollout-epochs", type=int, default=0,
                    help="epochs of full-horizon rollout training after the one-step stage; "
                         "required for the branch term, which needs trajectories")
    ap.add_argument("--branch-weight", type=float, default=0.0,
                    help="weight on branch separation over the rollout (0 = off). Scale-matched "
                         "to the state term on the stage's first batch, then multiplied by this")
    ap.add_argument("--states-per-batch", type=int, default=8)
    ap.add_argument("--hard-contacts", action="store_true",
                    help="feed back rounded contact flags instead of the probability")
    ap.add_argument("--no-orthonormalise", action="store_true",
                    help="skip re-projecting the rotation after each step (the old behaviour)")
    ap.add_argument("--cw-data", default=None,
                    help="contact-window branch groups (jenga_cw_data.py); their transitions join "
                         "the one-step pool. Off = the original D0 recipe, unchanged")
    ap.add_argument("--cw-loss", choices=("none", "branch", "intervention"), default="none",
                    help="extra objective on short unrolls of the contact-window branch groups: "
                         "branch = the existing pairwise-distance branch loss (D1), intervention "
                         "= CoCo-inspired intervention consistency (D2)")
    ap.add_argument("--cw-groups", type=int, default=32,
                    help="branch groups unrolled per one-step batch for the extra objective")
    ap.add_argument("--cw-weight", type=float, default=1.0,
                    help="weight on the extra objective; its terms are already normalised to O(1)")
    ap.add_argument("--quiet-quantile", type=float, default=0.5,
                    help="real intervention effects below this quantile count as 'no change'")
    ap.add_argument("--fast-branch", action="store_true",
                    help="batched branch_terms in the contact-window objective (speed only)")
    ap.add_argument("--compile", choices=("none", "inductor"), default="none",
                    help="torch.compile the short unroll of the contact-window objective. NOT "
                         "bit-identical: per-step rounding (~1e-7 to 1e-4 relative in gradients) "
                         "is amplified by Adam, so a compiled run does not reproduce an uncompiled "
                         "one. Never mix it with uncompiled runs in one arm. (The cudagraphs "
                         "backend was removed: it produced wrong gradients on this model.)")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="stop each epoch after this many batches (benchmarking only)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.cw_loss != "none" and not args.cw_data:
        ap.error("--cw-loss needs --cw-data")

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

    cw = None
    if args.cw_data:
        cw_start, cw_acts, cw_traj, cw_sources = load_cw(args.cw_data, val_ids)
        g, b, h = cw_acts.shape[:3]
        previous = torch.cat([cw_start[:, None, None].expand(g, b, 1, 61), cw_traj[:, :, :-1]], 2)
        cw = {"now": previous.reshape(-1, 61), "act": cw_acts.reshape(-1, 4),
              "next": cw_traj.reshape(-1, 61), "start": cw_start, "acts": cw_acts,
              "traj": cw_traj, "groups": g, "branches": b, "horizon": h}
        cont = torch.tensor(CONTINUOUS)
        effect = (cw_traj[:, 1:, :, CONTINUOUS] - cw_traj[:, :1, :, CONTINUOUS])
        # Scale of a real intervention effect per dimension, floored so dimensions that never
        # respond cannot blow up the normalisation.
        diff_scale = torch.maximum(effect.reshape(-1, len(CONTINUOUS)).std(0),
                                   1e-3 * state_scale[cont])
        end_effect = (effect[:, :, -1] / diff_scale).norm(dim=-1)
        cw["diff_scale"] = diff_scale.to(device)
        cw["cont_scale"] = state_scale[cont].to(device)
        cw["quiet_tau"] = float(torch.quantile(end_effect.reshape(-1).float(),
                                               args.quiet_quantile))
        print(f"contact-window data: {g} branch points x {b} branches x {h} steps = "
              f"{len(cw['now'])} extra transitions (loss: {args.cw_loss}, quiet tau "
              f"{cw['quiet_tau']:.3f})", flush=True)

    moe = args.model == "moe"
    if moe:
        model = StepGraphMoE(args.hidden, args.rounds, args.experts)
    elif args.model == "mlp":
        model = StepMLP()
    else:
        model = StepGraphNet(args.hidden, args.rounds)
    model = model.to(device)
    model.orthonormalise = not args.no_orthonormalise
    model.hard_contacts = args.hard_contacts
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model {args.model}: {parameters:,} parameters, orthonormalise "
          f"{model.orthonormalise}, hard contacts {model.hard_contacts}", flush=True)
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
        step_err, roll_err, predicted_flags, true_flags, ratios = [], [], [], [], []
        with torch.no_grad():
            for batch_start in range(0, len(val_states), 32):
                batch = val_states[batch_start:batch_start + 32]
                rows = np.concatenate(batch)
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
                _, ratio = branch_terms(predicted, truth, [len(g) for g in batch], state_scale_d)
                if ratio is not None:
                    ratios.append(ratio)
                del predicted, truth
        model.train()
        predicted_flags = np.concatenate(predicted_flags).any(-1)
        true_flags = np.concatenate(true_flags).any(-1)
        return {"one_step_mse": float(np.mean(step_err)),
                "rollout_state_error": float(np.mean(roll_err)),
                "branch_ratio": (float(np.mean(ratios)) if ratios else None),
                "topple_recall": (float(predicted_flags[true_flags].mean())
                                  if true_flags.any() else None),
                "topple_false_rate": (float(predicted_flags[~true_flags].mean())
                                      if (~true_flags).any() else None)}

    def plain_unroll(start_state, actions):
        return rollout(model, start_state, actions, delta_scale)

    unroll = plain_unroll
    if args.compile != "none":
        unroll = torch.compile(plain_unroll, backend=args.compile, fullgraph=False)

    def cw_objective():
        """The extra objective on a random batch of contact-window branch groups."""
        pick = torch.from_numpy(rng.choice(cw["groups"], args.cw_groups, replace=False))
        g, b, h = len(pick), cw["branches"], cw["horizon"]
        start_state = cw["start"][pick].to(device).repeat_interleave(b, 0)
        predicted = unroll(start_state, cw["acts"][pick].reshape(g * b, h, 4).to(device))
        truth = cw["traj"][pick].to(device)
        if args.cw_loss == "branch":
            if args.fast_branch:
                loss, _ = branch_terms_batched(predicted, truth.reshape(g * b, h, 61), b,
                                               state_scale_d)
            else:
                loss, _ = branch_terms(predicted, truth.reshape(g * b, h, 61), [b] * g,
                                       state_scale_d)
            return loss, {"branch": float(loss.detach())}
        terms = intervention_terms(predicted.reshape(g, b, h, 61), truth, cw["cont_scale"],
                                   cw["diff_scale"], cw["quiet_tau"])
        return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}

    history = []
    n_orig = len(train_transitions)
    for epoch in range(args.epochs):
        if cw is None:
            rng.shuffle(train_transitions)
        else:
            order = rng.permutation(n_orig + len(cw["now"]))
        start, running, extra = time.time(), [], []
        total = n_orig if cw is None else n_orig + len(cw["now"])
        batches_done = 0
        for first in range(0, total, args.batch):
            if args.max_batches is not None and batches_done >= args.max_batches:
                break
            batches_done += 1
            if cw is None:
                rows, steps = train_transitions[first:first + args.batch].T
                state = corrupt(traj[rows, steps].to(device))
                nxt = traj[rows, steps + 1].to(device)
                action = acts[rows, steps].to(device)
            else:
                chosen = order[first:first + args.batch]
                orig = chosen[chosen < n_orig]
                new = torch.from_numpy(chosen[chosen >= n_orig] - n_orig)
                rows, steps = train_transitions[orig].T
                state = corrupt(torch.cat([traj[rows, steps], cw["now"][new]]).to(device))
                nxt = torch.cat([traj[rows, steps + 1], cw["next"][new]]).to(device)
                action = torch.cat([acts[rows, steps], cw["act"][new]]).to(device)
            loss, mse, bce = one_step_loss(state, action, nxt)
            if cw is not None and args.cw_loss != "none":
                objective, parts = cw_objective()
                loss = loss + args.cw_weight * objective
                extra.append(parts)
            optimiser.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
            running.append((float(mse.detach()), float(bce.detach())))
        batch_seconds = time.time() - start
        validation = evaluate()
        if extra:
            validation["cw_train_terms"] = {k: float(np.mean([e[k] for e in extra]))
                                            for k in extra[0]}
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(running, axis=0)[0]),
                        "train_bce": float(np.mean(running, axis=0)[1]),
                        "validation": validation, "seconds": time.time() - start,
                        "batch_seconds": batch_seconds, "batches": batches_done})
        print(f"epoch {epoch + 1}/{args.epochs}: train mse {history[-1]['train_mse']:.4f} "
              f"val {validation} ({time.time() - start:.0f}s)", flush=True)

    # ---- Optional rollout stage: the branch term needs trajectories, not single steps.
    rollout_history = []
    if args.rollout_epochs:
        order = np.arange(len(train_states))
        weights = None
        for epoch in range(args.rollout_epochs):
            rng.shuffle(order)
            start, running = time.time(), []
            for first in range(0, len(order), args.states_per_batch):
                batch = [train_states[i] for i in order[first:first + args.states_per_batch]]
                rows = np.concatenate(batch)
                gates = [] if moe else None
                predicted = rollout(model, traj[rows, 0].to(device), acts[rows].to(device),
                                    delta_scale, collect=gates)
                truth = traj[rows, 1:].to(device)
                terms = {"state": (((predicted[..., :45] - truth[..., :45])
                                    / state_scale_d[:45]) ** 2).mean(),
                         "contact": F.binary_cross_entropy(
                             predicted[..., 45:57].clamp(1e-5, 1 - 1e-5), truth[..., 45:57])}
                if args.branch_weight:
                    terms["branch"], _ = branch_terms(predicted, truth,
                                                      [len(g) for g in batch], state_scale_d)
                if moe:
                    terms["balance"] = load_balance(torch.stack(gates, dim=1))
                if weights is None:
                    # Scale-match to the state term once, then apply the chosen multipliers.
                    weights = {k: float(terms["state"].detach()) / max(float(v.detach()), 1e-8)
                               for k, v in terms.items()}
                    weights["state"] = 1.0
                    if "branch" in weights:
                        weights["branch"] *= args.branch_weight
                    if "balance" in weights:
                        weights["balance"] = args.balance_weight
                loss = sum(weights[k] * v for k, v in terms.items())
                optimiser.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
                running.append({k: float(v.detach()) for k, v in terms.items()})
            validation = evaluate()
            rollout_history.append({"epoch": epoch + 1, "weights": weights,
                                    "train": {k: float(np.mean([r[k] for r in running]))
                                              for k in running[0]},
                                    "validation": validation, "seconds": time.time() - start})
            print(f"rollout epoch {epoch + 1}/{args.rollout_epochs}: "
                  f"{rollout_history[-1]['train']} val {validation} "
                  f"({time.time() - start:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "hidden": args.hidden, "rounds": args.rounds,
                "kind": args.model, "experts": args.experts if moe else 1,
                "parameters": parameters,
                "substeps": substeps, "seed": args.seed, "noise": args.noise,
                "orthonormalise": model.orthonormalise, "hard_contacts": model.hard_contacts,
                "branch_weight": args.branch_weight, "rollout_epochs": args.rollout_epochs,
                "cw_data": args.cw_data, "cw_loss": args.cw_loss, "cw_groups": args.cw_groups,
                "cw_weight": args.cw_weight,
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
         "model": args.model, "branch_weight": args.branch_weight,
         "cw_data": args.cw_data, "cw_loss": args.cw_loss, "cw_groups": args.cw_groups,
         "cw_weight": args.cw_weight, "quiet_quantile": args.quiet_quantile,
         "orthonormalise": model.orthonormalise, "hard_contacts": model.hard_contacts,
         "states": len(groups), "history": history, "rollout_history": rollout_history,
         "output": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
