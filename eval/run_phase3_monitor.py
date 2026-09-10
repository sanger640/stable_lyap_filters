"""
Phase 3 monitor, done properly: learned model, corrected horizon, probe families compared.

Everything earlier either used the true simulator or ran at a horizon too short for the failure
to express itself in the state (the block commits at ~step 133 and takes 164 steps to fall, so
a 200-step rollout measures a block still mid-air). Both are fixed here:

  * the model is TRAINED and ROLLED OUT at the full horizon
  * probes are compared as a family, not one-off

PROBE FAMILIES. The key algebraic point is that

    a + c*a  ==  a*(1 + c)

so "multiplicative shared" is just "additive shared along the direction v = a". There is only
ONE mechanism -- perturb along a chosen direction with a single scalar -- and the only question
is which v. That matters for transfer: Jenga's action is an absolute EE position, where
multiplying is geometrically meaningless (origin-dependent), but perturbing along a chosen
direction is perfectly well defined.

    per-step        a + eps*thr*z,  z in R^T        <- what the deviator agent does now
    shared-const    a + eps*thr*z*1, z scalar       <- one coherent offset everywhere
    shared-action   a + eps*z*a,     z scalar       <- along the action itself (== scaling)
    shared-envelope a + eps*thr*z*sign(a)           <- coherent, but only where force exists

Fawzi et al. (NeurIPS 2016), Thm 1 predicts the per-step family needs ~sqrt(T) times the
magnitude to reach the same boundary, since a random direction has expected squared overlap 1/T
with the boundary normal. T=450 here, so ~21x.
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
sys.path.insert(0, str(ROOT / "eval"))

import tipping_block as tb                                        # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from gtf import gtf_rollout_loss, gtf_rollout_loss_latent         # noqa: E402
from run_phase3_block import auc, metrics                         # noqa: E402

OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
OBS_DIM, ACT_DIM = 2, 1
T_ON, T_OFF = 10, 60


def make_data(n_traj, thr, n, seed=0):
    rng = np.random.default_rng(seed)
    S, A = [], []
    for _ in range(n_traj):
        act = tb.random_push(rng, n, thr)
        st, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        S.append(st[:-1]); A.append(act[:, None])
    S = torch.from_numpy(np.stack(S)).double()
    A = torch.from_numpy(np.stack(A)).double()
    mu, sd = S.reshape(-1, OBS_DIM).mean(0), S.reshape(-1, OBS_DIM).std(0)
    amu, asd = A.reshape(-1, ACT_DIM).mean(0), A.reshape(-1, ACT_DIM).std(0)
    return (S - mu) / sd, (A - amu) / asd, mu, sd, amu, asd


def train(S, A, epochs, H, d, seq_len=40, batch=128, lr=3e-3, alpha=0.02, seed=0):
    torch.manual_seed(seed)
    model = ShPLRNN(d=d, H=H, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_traj, n, _ = S.shape
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
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"      ep {ep:4d}  loss {tot/nb:.3e}", flush=True)
    return model


@torch.no_grad()
def rollout(model, act_norm, s0_norm, T):
    z = model.lift(s0_norm)
    zs = []
    for t in range(T):
        z = model(z, act_norm[:, t])
        zs.append(z.clone())
    return torch.stack(zs, 1)


def probes(kind, a, thr, eps, NP, rng):
    """All four families reduce to a + (scalar or vector) * direction."""
    T = len(a)
    if kind == "per-step":
        return a[None] + thr * eps * rng.standard_normal((NP, T))
    z = rng.standard_normal((NP, 1))
    if kind == "shared-const":
        return a[None] + thr * eps * z
    if kind == "shared-action":                       # == a * (1 + eps*z)
        return a[None] + eps * z * a[None]
    if kind == "shared-envelope":
        return a[None] + thr * eps * z * np.sign(a)[None]
    raise ValueError(kind)


def margin_of(act, s_max=0.5, coarse=0.02):
    base = bool(tb.simulate(act)[1][-1])
    for i in range(1, int(s_max / coarse) + 1):
        for sg in (+1, -1):
            s = 1.0 + sg * i * coarse
            if s <= 0:
                continue
            if bool(tb.simulate(act * s)[1][-1]) != base:
                lo, hi = 1.0 + sg * (i - 1) * coarse, s
                for _ in range(18):
                    m = 0.5 * (lo + hi)
                    if bool(tb.simulate(act * m)[1][-1]) != base:
                        hi = m
                    else:
                        lo = m
                return abs(0.5 * (lo + hi) - 1.0)
    return s_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n-traj", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--n-perturb", type=int, default=32)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.10])
    ap.add_argument("--near", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    t0 = time.time()
    n = args.steps
    ckpt = OUT / f"monitor_model_T{n}.pt"

    thr = tb.topple_threshold(n, T_ON, T_OFF)
    S, A, mu, sd, amu, asd = make_data(args.n_traj, thr, n, seed=args.seed)
    if ckpt.exists():
        model = ShPLRNN(d=args.d, H=args.H, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
        model.load_state_dict(torch.load(ckpt)); print(f"loaded {ckpt}\n", flush=True)
    else:
        print(f"training at horizon {n} ...", flush=True)
        model = train(S, A, args.epochs, args.H, args.d, seed=args.seed)
        torch.save(model.state_dict(), ckpt)

    rng = np.random.default_rng(args.seed + 1)
    acts = np.stack([tb.random_push(rng, n, thr) for _ in range(args.n_test)])
    print("computing ground truth ...", flush=True)
    margin = np.array([margin_of(a) for a in acts])
    outcome = np.array([bool(tb.simulate(a)[1][-1]) for a in acts])
    near = margin < args.near
    print(f"  {args.n_test} actions: {int(outcome.sum())} topple, {int(near.sum())} near\n")

    # model fidelity at the full horizon -- an autoregressive 450-step rollout is a big ask
    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    an = ((torch.from_numpy(acts[:, :, None]).double() - amu) / asd)
    Z = rollout(model, an, s0.expand(len(acts), OBS_DIM), n)
    pred_theta = (model.observe(Z)[:, :, 0] * sd[0] + mu[0]).numpy()
    true_theta = np.stack([tb.simulate(a)[0][1:n + 1, 0] for a in acts])
    print(f"model fidelity: finite={np.isfinite(pred_theta).all()}  "
          f"corr(theta) {np.corrcoef(pred_theta.ravel(), true_theta.ravel())[0,1]:+.3f}  "
          f"RMSE {np.sqrt(((pred_theta-true_theta)**2).mean()):.3f} rad\n", flush=True)

    rows = {}
    for eps in args.eps:
        print(f"=== eps={eps}  (LEARNED MODEL, horizon {n}) ===")
        print(f"{'probe family':<18}{'AUC outcome':>13}{'AUC proximity':>15}{'prox F1':>9}"
              f"{'out F1':>8}")
        for kind in ("per-step", "shared-const", "shared-action", "shared-envelope"):
            sc = []
            for a in acts:
                P = probes(kind, a, thr, eps, args.n_perturb, rng)
                pn = ((torch.from_numpy(P[:, :, None]).double() - amu) / asd)
                Zp = rollout(model, pn, s0.expand(len(P), OBS_DIM), n)
                d = Zp[:, -1].norm(dim=-1)               # scalar per rollout
                sc.append(float(d.std()))                # divergence std
            sc = np.array(sc)
            mo, mp = metrics(sc, outcome), metrics(sc, near)
            rows[f"{eps}|{kind}"] = {"outcome": mo, "proximity": mp}
            print(f"{kind:<18}{mo['auc']:>13.3f}{mp['auc']:>15.3f}{mp['f1']:>9.3f}"
                  f"{mo['f1']:>8.3f}", flush=True)
        print()

    json.dump({"threshold": thr, "args": vars(args), "rows": rows},
              open(OUT / f"monitor_{args.tag}.json", "w"), indent=1)
    print(f"-> {OUT / f'monitor_{args.tag}.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
