"""
Is divergence a good UNIVERSAL proxy for failure?

The previous test compared divergence against `max|theta|` and `net impulse`, and both beat it.
That comparison was unfair in a specific way: those two are PRIVILEGED. `max|theta|` only works
because we know which state coordinate means "tilt", and `net impulse` only because we know the
action is a force. In a DINO latent space neither is available -- no coordinate means anything,
and there is no failure predicate to read off. That is the entire reason divergence is
attractive: it needs no labels, no failure definition, and no interpretable state.

So the honest question is not "does divergence beat an oracle" but:

    among statistics computable from ONLY (latent rollouts, perturbed actions) -- no knowledge
    of what any dimension means, no failure predicate -- is divergence the best one, and is
    what it achieves usable?

Every UNIVERSAL score below is computed from latent vectors alone via norms and differences.
None would need a single change to run on DINO latents. The privileged scores are kept only as
a reference line for how much is being given up.

The one to watch is `latent displacement`: ||z_T - z_0|| for the UNPERTURBED rollout. It is
fully universal -- it needs no perturbations at all and no idea what the dimensions mean -- and
it is the interpretation-free stand-in for "roll it out and look".
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
from run_phase3_block import (ACT_DIM, N_STEPS, OBS_DIM, auc, make_data,  # noqa: E402
                              metrics, train)

OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "universal_model.pt"


@torch.no_grad()
def latent_rollout(model, act_norm, s0_norm, T):
    """Full LATENT trajectory (not the readout) -- what a monitor on DINO features would see."""
    z = model.lift(s0_norm)
    zs = []
    for t in range(T):
        z = model(z, act_norm[:, t])
        zs.append(z.clone())
    return torch.stack(zs, 1)                             # (B, T, d)


def universal_scores(Z, eps_start):
    """Every statistic here uses only norms and differences of latent vectors.

    Z: (1 + n_pert, T, d), row 0 = unperturbed. Nothing below refers to a named coordinate, a
    failure predicate, or the action's physical meaning."""
    z0, zp = Z[0], Z[1:]
    end_o, end_p = z0[-1], zp[:, -1]
    d_end = (end_p - end_o).norm(dim=-1)
    d_start = (zp[:, 0] - z0[0]).norm(dim=-1).clamp_min(1e-12)
    centroid = end_p.mean(0)
    return {
        # --- perturbation-based (the monitor family) ---
        "divergence max": float(d_end.max()),
        "divergence mean": float(d_end.mean()),
        "divergence spread": float((end_p - centroid).norm(dim=-1).mean()),
        "FTLE ratio": float(torch.log(d_end / d_start).max()),
        # --- single-rollout, no perturbations at all ---
        "latent displacement": float((z0[-1] - z0[0]).norm()),
        "latent path length": float((z0[1:] - z0[:-1]).norm(dim=-1).sum()),
        "latent max speed": float((z0[1:] - z0[:-1]).norm(dim=-1).max()),
        "latent final norm": float(z0[-1].norm()),
        # --- SHAPE of the perturbed endpoint cloud, not its size ---------------------------
        # The threshold problem: every score above is in arbitrary latent units, so its cut has
        # to be calibrated. This one is a shape test instead. If the perturbations STRADDLE a
        # boundary their endpoints split into two clusters (some topple, some do not); if the
        # action is far from any boundary they form one blob. The ratio below is scale-free --
        # it divides out the latent units -- so ~1 is a meaningful cut rather than a tuned one.
        "endpoint bimodality": _bimodality(end_p),
    }


def _bimodality(pts, iters=25):
    """2-means separation / within-cluster spread. Scale-free: >1 means the two clusters are
    further apart than they are wide, i.e. a genuine split rather than one elongated blob."""
    if len(pts) < 4:
        return 0.0
    x = pts - pts.mean(0)
    # initialise on the principal direction so the split is not seed-dependent
    u = torch.linalg.svd(x, full_matrices=False)[2][0]
    proj = x @ u
    c = torch.stack([pts[proj <= 0].mean(0) if (proj <= 0).any() else pts[0],
                     pts[proj > 0].mean(0) if (proj > 0).any() else pts[-1]])
    for _ in range(iters):
        lab = torch.cdist(pts, c).argmin(1)
        for k in (0, 1):
            if (lab == k).any():
                c[k] = pts[lab == k].mean(0)
    sep = float((c[0] - c[1]).norm())
    within = float(torch.stack([(pts[lab == k] - c[k]).norm(dim=-1).mean()
                                for k in (0, 1) if (lab == k).any()]).mean())
    return sep / (within + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n-traj", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-perturb", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--horizons", type=int, nargs="+", default=[60, 80, 100])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    thr = tb.topple_threshold(N_STEPS, 10, 60)
    S, A, mu, sd, amu, asd = make_data(args.n_traj, thr, seed=args.seed)
    if CKPT.exists():
        model = ShPLRNN(d=args.d, H=args.H, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
        model.load_state_dict(torch.load(CKPT)); print(f"loaded {CKPT}\n", flush=True)
    else:
        print(f"training ({args.epochs} epochs) ...", flush=True)
        model = train(S, A, args.epochs, args.H, args.d, seed=args.seed)
        torch.save(model.state_dict(), CKPT)

    rng = np.random.default_rng(args.seed + 1)
    acts = np.stack([tb.random_push(rng, N_STEPS, thr) for _ in range(args.n_test)])
    label = np.array([bool(tb.simulate(a, dt=tb.DT_DEFAULT)[1][-1]) for a in acts])
    print(f"test: {args.n_test} pushes -> {int(label.sum())} topple / "
          f"{int((~label).sum())} survive\n")

    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    out = {}
    for T in args.horizons:
        acc = None
        priv_theta = []
        for a in acts:
            base = a[:T]
            pert = base[None] + thr * args.eps * rng.standard_normal((args.n_perturb, T))
            allact = np.concatenate([base[None], pert])
            actn = ((torch.from_numpy(allact[..., None]).double() - amu) / asd)
            Z = latent_rollout(model, actn, s0.expand(len(allact), OBS_DIM), T)
            sc = universal_scores(Z, thr * args.eps)
            if acc is None:
                acc = {k: [] for k in sc}
            for k, v in sc.items():
                acc[k].append(v)
            obs = model.observe(Z[0])
            priv_theta.append(float((obs[:, 0] * sd[0] + mu[0]).abs().max()))

        res = {k: metrics(v, label) for k, v in acc.items()}
        res["[privileged] max|theta|"] = metrics(priv_theta, label)
        res["[privileged] net impulse"] = metrics(np.abs(acts[:, :T].sum(1)), label)
        out[T] = res

        print(f"=== horizon T={T} " + "=" * 46)
        print(f"{'universal statistic':<26}{'AUC':>7}{'prec':>8}{'recall':>8}{'F1':>7}{'acc':>7}")
        for k in sorted([k for k in res if not k.startswith("[")],
                        key=lambda k: -res[k]["auc"]):
            m = res[k]
            print(f"{k:<26}{m['auc']:>7.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                  f"{m['f1']:>7.3f}{m['accuracy']:>7.3f}")
        print(f"{'-- privileged (not available on DINO latents) --':<26}")
        for k in [k for k in res if k.startswith("[")]:
            m = res[k]
            print(f"{k:<26}{m['auc']:>7.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                  f"{m['f1']:>7.3f}{m['accuracy']:>7.3f}")
        print(flush=True)

    json.dump({"threshold": thr, "args": vars(args),
               "rows": {str(k): v for k, v in out.items()}},
              open(OUT / "universal.json", "w"), indent=1)
    print(f"-> {OUT / 'universal.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
