"""
Phase 3 — the actual monitor question, on a system where it is finally well posed.

Given an action sequence, does perturbation-divergence IN THE LEARNED MODEL predict whether the
TRUE system topples? The tipping block makes this answerable in a way the bouncing ball never
was: an exact boundary in action space (push amplitude 0.6311 at the default shape), an
absorbing failure, and a 164-step gap between the block committing to fall and it being
visibly over.

Two controls decide whether any positive result means anything:

  DIRECT PREDICTION.  The model can simply be rolled out and asked "does theta exceed alpha?".
      If that beats divergence, the monitor is unnecessary machinery -- you would just use the
      model. This is the control the Jenga setup cannot easily run (failure is not a single
      observable there) and it is the one most likely to deflate the approach.

  ORACLE DIVERGENCE.  The same divergence computed on the TRUE simulator. If the statistic
      fails there too, no model can rescue it -- the same reasoning that killed max|local
      lambda| in Phase 2.

EARLY WARNING is the point. The monitor is scored at horizons BEFORE the block visibly falls,
because a detector that only fires once the block is on the floor is worthless.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "geometry", "models", "training")]

import tipping_block as tb                                        # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from gtf import gtf_rollout_loss, gtf_rollout_loss_latent        # noqa: E402

OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
OBS_DIM, ACT_DIM = 2, 1
T_ON, T_OFF, N_STEPS = 10, 60, 200


def auc(score, y):
    o = np.argsort(score); y = np.asarray(y)[o]
    r = np.arange(1, len(y) + 1)
    p, q = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - p * (p + 1) / 2) / (p * q)) if p and q else float("nan")


def make_data(n_traj, thr, seed=0, n=N_STEPS):
    """Trajectories from random pushes spanning safe and toppling.

    The amplitude range deliberately straddles the threshold so the model sees BOTH outcomes;
    a model trained only on safe pushes could not represent the failure at all."""
    rng = np.random.default_rng(seed)
    S, A = [], []
    for _ in range(n_traj):
        act = tb.random_push(rng, n, thr)
        st, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        S.append(st[:-1]); A.append(act[:, None])
    S = torch.from_numpy(np.stack(S)).double()          # (n_traj, n, 2)
    A = torch.from_numpy(np.stack(A)).double()          # (n_traj, n, 1)
    mu, sd = S.reshape(-1, OBS_DIM).mean(0), S.reshape(-1, OBS_DIM).std(0)
    amu, asd = A.reshape(-1, ACT_DIM).mean(0), A.reshape(-1, ACT_DIM).std(0)
    return (S - mu) / sd, (A - amu) / asd, mu, sd, amu, asd


def train(S, A, epochs, H, d, seq_len=40, batch=128, lr=3e-3, alpha=0.02, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = ShPLRNN(d=d, H=H, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_traj, n, _ = S.shape
    # d > obs_dim means the auxiliary latents are never forced, only the observed readout is
    loss_fn = gtf_rollout_loss if d == OBS_DIM else gtf_rollout_loss_latent
    for ep in range(epochs):
        tot = nb = 0
        for _ in range(20):
            ti = torch.randint(0, n_traj, (batch,))
            t0 = torch.randint(0, n - seq_len - 1, (batch,))
            seq = torch.stack([S[a, b:b + seq_len] for a, b in zip(ti, t0)])
            act = torch.stack([A[a, b:b + seq_len] for a, b in zip(ti, t0)])
            loss = loss_fn(model, seq, alpha, action_seq=act)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"      ep {ep:4d}  loss {tot/nb:.3e}", flush=True)
    return model


@torch.no_grad()
def model_rollout(model, act_norm, s0_norm, T):
    """Roll the model under a batch of (possibly perturbed) action sequences."""
    z = model.lift(s0_norm)
    outs = []
    for t in range(T):
        z = model(z, act_norm[:, t])
        outs.append(model.observe(z))
    return torch.stack(outs, 1)                          # (B, T, 2)


def true_rollout(acts, T):
    out = np.empty((len(acts), T, 2))
    for i, a in enumerate(acts):
        st, _ = tb.simulate(a[:T], dt=tb.DT_DEFAULT)
        out[i] = st[:T]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n-traj", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-perturb", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.02, help="action noise, fraction of thr")
    ap.add_argument("--horizons", type=int, nargs="+", default=[40, 60, 80, 100, 133])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()

    thr = tb.topple_threshold(N_STEPS, T_ON, T_OFF)
    print(f"topple threshold = {thr:.6f}   (static lift-off {tb.critical_force():.4f})\n")

    S, A, mu, sd, amu, asd = make_data(args.n_traj, thr, seed=args.seed)
    print(f"training shPLRNN (d={args.d}, H={args.H}, {args.n_traj} trajectories) ...",
          flush=True)
    model = train(S, A, args.epochs, args.H, args.d, seed=args.seed)

    # ---- test set: amplitudes straddling the boundary, so the task is genuinely hard --------
    rng = np.random.default_rng(args.seed + 1)
    acts_test, label = [], []
    while len(acts_test) < args.n_test:
        act = tb.random_push(rng, N_STEPS, thr)
        fell = bool(tb.simulate(act, dt=tb.DT_DEFAULT)[1][-1])
        acts_test.append(act); label.append(fell)
    acts_test = np.stack(acts_test); label = np.array(label)
    print(f"test set: {args.n_test} multi-pulse pushes -> "
          f"{int(label.sum())} topple, {int((~label).sum())} survive")
    peak = np.abs(acts_test).max(1); imp = np.abs(acts_test.sum(1))
    print(f"  giveaway baselines from the ACTION alone: peak force {auc(peak, label):.3f}, "
          f"net impulse {auc(imp, label):.3f}")
    print("  (a single-amplitude push would give 1.000 for both -- the task must beat these)\n")

    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    rows = {}
    print(f"{'T':>5}{'visible?':>10}{'MODEL diverg':>14}{'MODEL max|θ|':>14}"
          f"{'TRUE diverg':>13}")
    for T in args.horizons:
        # what fraction have VISIBLY toppled by T -- an honest early-warning check
        vis = np.array([np.abs(tb.simulate(a[:T], dt=tb.DT_DEFAULT)[0][:, 0]).max()
                        >= tb.ALPHA_DEFAULT for a in acts_test]).mean()

        div_m, maxth_m, div_t = [], [], []
        for a in acts_test:
            base = a[:T]
            pert = base[None] + thr * args.eps * rng.standard_normal((args.n_perturb, T))
            allact = np.concatenate([base[None], pert])
            actn = ((torch.from_numpy(allact[..., None]).double() - amu) / asd)
            traj = model_rollout(model, actn, s0.expand(len(allact), OBS_DIM), T)
            d = (traj[1:, -1] - traj[0, -1]).norm(dim=-1)
            div_m.append(float(d.max()))
            maxth_m.append(float((traj[0, :, 0] * sd[0] + mu[0]).abs().max()))
            tr = true_rollout(allact, T)
            div_t.append(float(np.linalg.norm(tr[1:, -1] - tr[0, -1], axis=-1).max()))

        rows[T] = {"visible_frac": float(vis),
                   "auc_model_divergence": auc(np.array(div_m), label),
                   "auc_model_max_theta": auc(np.array(maxth_m), label),
                   "auc_true_divergence": auc(np.array(div_t), label)}
        r = rows[T]
        print(f"{T:>5}{100*vis:>9.0f}%{r['auc_model_divergence']:>14.3f}"
              f"{r['auc_model_max_theta']:>14.3f}{r['auc_true_divergence']:>13.3f}", flush=True)

    print("\n'visible?' = fraction of test pushes whose block has already passed alpha by T.")
    print("A monitor is only interesting where that is LOW -- otherwise it is reporting the")
    print("obvious. 'MODEL max|θ|' is the deflating control: just roll the model out and look.")
    json.dump({"threshold": thr, "args": vars(args), "rows": rows},
              open(OUT / f"monitor_{args.tag}.json", "w"), indent=1)
    print(f"\n-> {OUT / f'monitor_{args.tag}.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
