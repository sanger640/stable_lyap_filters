"""
Does divergence detect PROXIMITY to the failure boundary? (The question that was being asked.)

Earlier phases scored every statistic on "will this action topple the block". On that task
divergence lost to latent displacement, and I described the reason as a defect: divergence
measures |distance to the boundary| rather than which SIDE of it you are on.

That framing was wrong for the actual goal. A monitor that only fires once an action is already
unsafe leaves no margin -- you want to know you are APPROACHING the boundary while still on the
safe side. A statistic that peaks at the boundary is exactly the right instrument for that, so
divergence was being graded on the wrong task.

MARGIN, the ground truth here: for an action a, scale it by s and find the s nearest 1 at which
the outcome flips. margin = |s* - 1| is then "how much could this action be scaled before the
outcome changes" -- a genuine distance to the failure boundary in ACTION space, defined without
reference to any state coordinate.

Two tasks are scored side by side, on the same actions and the same scores:

    OUTCOME    will it topple?          (the earlier task)
    PROXIMITY  is the margin small?     (the task actually wanted)

A statistic can be good at one and bad at the other, and that is the whole point.
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
from phase3_universal import latent_rollout, universal_scores     # noqa: E402

OUT = ROOT / "results" / "phase3"
CKPT = OUT / "universal_model.pt"


def action_margin(act, s_max=0.6, coarse=0.02):
    """Smallest |s - 1| such that scaling the action by s flips the outcome.

    Scans outward from s = 1 in both directions, then bisects. Returns s_max if no flip is
    found within the range -- i.e. the action is comfortably far from the boundary."""
    base = bool(tb.simulate(act, dt=tb.DT_DEFAULT)[1][-1])
    n = int(s_max / coarse)
    for i in range(1, n + 1):
        for sign in (+1, -1):
            s = 1.0 + sign * i * coarse
            if s <= 0:
                continue
            if bool(tb.simulate(act * s, dt=tb.DT_DEFAULT)[1][-1]) != base:
                lo, hi = 1.0 + sign * (i - 1) * coarse, s      # bisect for the flip
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if bool(tb.simulate(act * mid, dt=tb.DT_DEFAULT)[1][-1]) != base:
                        hi = mid
                    else:
                        lo = mid
                return abs(0.5 * (lo + hi) - 1.0)
    return s_max


def action_margin_random(act, thr, rng, n_dir=12, r_max=0.5, n_grid=7, n_bisect=8):
    """Distance to the boundary along the SAME directions the monitor perturbs in.

    `action_margin` above scales the whole action up and down -- a 1-D slice through a
    T-dimensional action space. But divergence probes RANDOM directions, so scoring it against
    a scaling margin grades it on geometry it never samples. This measures the smallest random
    perturbation that flips the outcome, which is the quantity divergence is actually sensing.

    Parameterisation is matched to the monitor exactly: the monitor perturbs with
    thr * eps * N(0, I_T), so here a direction is a unit Gaussian rescaled to sqrt(T) and the
    radius r plays the role of eps. A margin of 0.1 therefore means "some 10%-sized
    perturbation flips this outcome", directly comparable to the eps sweep."""
    T = len(act)
    base = bool(tb.simulate(act, dt=tb.DT_DEFAULT)[1][-1])
    best = r_max
    grid = np.geomspace(0.01, r_max, n_grid)
    for _ in range(n_dir):
        g = rng.standard_normal(T)
        u = g / np.linalg.norm(g) * np.sqrt(T)          # same per-component scale as N(0,I)
        lo, hi = 0.0, None
        for r in grid:
            if r >= best:
                break                                   # cannot improve on the current best
            if bool(tb.simulate(act + thr * r * u, dt=tb.DT_DEFAULT)[1][-1]) != base:
                hi = r
                break
            lo = r
        if hi is not None:
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if bool(tb.simulate(act + thr * mid * u, dt=tb.DT_DEFAULT)[1][-1]) != base:
                    hi = mid
                else:
                    lo = mid
            best = min(best, hi)
    return best


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n-traj", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--n-perturb", type=int, default=32)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02])
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--near", type=float, default=0.10, help="margin below this = 'near'")
    ap.add_argument("--n-dir", type=int, default=12, help="random directions per margin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    thr = tb.topple_threshold(N_STEPS, 10, 60)
    S, A, mu, sd, amu, asd = make_data(args.n_traj, thr, seed=args.seed)
    if CKPT.exists():
        model = ShPLRNN(d=args.d, H=args.H, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
        model.load_state_dict(torch.load(CKPT)); print(f"loaded {CKPT}", flush=True)
    else:
        model = train(S, A, args.epochs, args.H, args.d, seed=args.seed)
        torch.save(model.state_dict(), CKPT)

    rng = np.random.default_rng(args.seed + 1)
    acts = np.stack([tb.random_push(rng, N_STEPS, thr) for _ in range(args.n_test)])
    print("computing action-space margins (ground truth for proximity) ...", flush=True)
    margin_scale = np.array([action_margin(a) for a in acts])
    mrng = np.random.default_rng(args.seed + 99)
    margin = np.array([action_margin_random(a, thr, mrng, n_dir=args.n_dir) for a in acts])
    outcome = np.array([bool(tb.simulate(a, dt=tb.DT_DEFAULT)[1][-1]) for a in acts])
    near = margin < args.near
    print(f"  scaling margin  : median {np.median(margin_scale):.3f}  "
          f"near(<{args.near}) {int((margin_scale < args.near).sum())}")
    print(f"  RANDOM-DIR margin: median {np.median(margin):.3f}  "
          f"near(<{args.near}) {int(near.sum())}   "
          f"(rank corr with scaling margin {spearman(margin, margin_scale):+.3f})")
    print(f"  {args.n_test} actions: {int(outcome.sum())} topple, "
          f"{int(near.sum())} within {args.near:.0%} of the boundary "
          f"(random-direction definition)\n", flush=True)

    T = args.horizon
    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    all_res = {}
    for eps in args.eps:
        acc = None
        for a in acts:
            base = a[:T]
            pert = base[None] + thr * eps * rng.standard_normal((args.n_perturb, T))
            allact = np.concatenate([base[None], pert])
            actn = ((torch.from_numpy(allact[..., None]).double() - amu) / asd)
            Z = latent_rollout(model, actn, s0.expand(len(allact), OBS_DIM), T)
            sc = universal_scores(Z, thr * eps)
            if acc is None:
                acc = {k: [] for k in sc}
            for k, v in sc.items():
                acc[k].append(v)
        res = {}
        for k, v in acc.items():
            v = np.array(v)
            res[k] = {"auc_outcome": auc(v, outcome), "auc_proximity": auc(v, near),
                      "spearman_vs_margin": spearman(v, margin), "prox": metrics(v, near)}
        all_res[eps] = res

        print(f"=== eps={eps:.2f} (probes a {eps:.0%} neighbourhood), T={T}, "
              f"PROXIMITY = margin < {args.near:.0%} ===")
        print(f"{'statistic':<24}{'AUC outcome':>13}{'AUC proximity':>15}{'rho vs margin':>15}")
        for k in sorted(res, key=lambda k: -res[k]["auc_proximity"]):
            r = res[k]
            print(f"{k:<24}{r['auc_outcome']:>13.3f}{r['auc_proximity']:>15.3f}"
                  f"{r['spearman_vs_margin']:>15.3f}")
        print(flush=True)

    # PREDICTION under test: eps sets the WIDTH of the boundary layer a perturbation probe can
    # reach, so proximity detection should peak where eps ~ the margin being asked about.
    print(f"=== proximity AUC vs eps (margin threshold = {args.near:.0%}) ===")
    keys = ["divergence mean", "divergence spread", "divergence max", "latent displacement"]
    print(f"{'eps':>7}" + "".join(f"{k:>20}" for k in keys))
    for eps in args.eps:
        print(f"{eps:>7.2f}" + "".join(f"{all_res[eps][k]['auc_proximity']:>20.3f}"
                                       for k in keys))

    json.dump({"threshold": thr, "args": vars(args),
               "results": {str(e): r for e, r in all_res.items()},
               "n_near": int(near.sum()), "n_topple": int(outcome.sum())},
              open(OUT / "margin.json", "w"), indent=1)
    print(f"\n-> {OUT / 'margin.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
