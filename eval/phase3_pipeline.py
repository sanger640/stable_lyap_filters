"""END-TO-END monitor: discover attractors automatically, then score by basin entropy.

Every component was validated in isolation; this assembles them and measures the result, so the
headline number is not stitched together from runs with different settings.

  OFFLINE  roll actions through the model, then keep rolling with the ACTION OFF so states
           settle. Keep the settled ones. Merge them at a swept distance; the attractor count
           is the PLATEAU -- the value stable across a range of merge distances. No k supplied.
  RUNTIME  perturb the action (coherent 'shared-action' probes), roll + settle each, assign
           every probe to its nearest attractor.
  SCORE    basin entropy S = -sum p_j ln p_j over the attractors reached.
  ALARM    S > 0  -- i.e. the probes did not all agree. Calibration-free by construction:
           S is built from a probability, so it has no units to normalise.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))
import tipping_block as tb                                                    # noqa: E402
from shplrnn import ShPLRNN                                                   # noqa: E402
from run_phase3_monitor import (ACT_DIM, OBS_DIM, T_ON, T_OFF, make_data,     # noqa: E402
                                margin_of, probes, rollout)
from run_phase3_block import auc, metrics                                     # noqa: E402
OUT = ROOT / "results" / "phase3"


def settle_latents(model, A, mu, sd, amu, asd, s0, n, settle):
    """Roll the action, then keep rolling with zero action so the state can come to rest."""
    long = np.concatenate([A, np.zeros((len(A), settle))], axis=1)
    pn = ((torch.from_numpy(long[:, :, None]).double() - amu) / asd)
    Z = rollout(model, pn, s0.expand(len(A), OBS_DIM), n + settle)
    return Z[:, -1].numpy(), (Z[:, -1] - Z[:, -2]).norm(dim=-1).numpy()


def merge(X, d):
    lab = -np.ones(len(X), int); g = 0
    for i in range(len(X)):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = g
        while stack:
            j = stack.pop()
            nb = np.where((lab < 0) & (np.linalg.norm(X - X[j], axis=1) < d))[0]
            lab[nb] = g; stack.extend(nb.tolist())
        g += 1
    return lab


def find_attractors(E, scale, fracs=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)):
    """The attractor count is the plateau: the count that survives the widest range of d."""
    counts = [(f, merge(E, f * scale).max() + 1) for f in fracs]
    runs, cur = [], [counts[0]]
    for prev, c in zip(counts, counts[1:]):
        if c[1] == prev[1]:
            cur.append(c)
        else:
            runs.append(cur); cur = [c]
    runs.append(cur)
    best = max(runs, key=len)
    d = np.median([f for f, _ in best]) * scale
    lab = merge(E, d)
    C = np.stack([E[lab == j].mean(0) for j in range(lab.max() + 1)])
    return C, d, counts, len(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--settle", type=int, default=400)
    ap.add_argument("--n-offline", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--n-perturb", type=int, default=48)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--near", type=float, default=0.10)
    args = ap.parse_args(); t0 = time.time()
    n, st = args.steps, args.settle
    thr = tb.topple_threshold(n, T_ON, T_OFF)
    _, _, mu, sd, amu, asd = make_data(40, thr, n, seed=0)
    model = ShPLRNN(d=4, H=128, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    model.load_state_dict(torch.load(OUT / f"monitor_model_T{n}.pt"))
    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    rng = np.random.default_rng(21)

    print("OFFLINE: discovering attractors (no k supplied) ...", flush=True)
    off = np.stack([tb.random_push(rng, n, thr) for _ in range(args.n_offline)])
    E, step = settle_latents(model, off, mu, sd, amu, asd, s0, n, st)
    scale = np.linalg.norm(E - E.mean(0), axis=1).mean()
    conv = step < 0.01 * scale
    C, d, counts, plat = find_attractors(E[conv], scale)
    print(f"  settled {conv.sum()}/{len(off)}   merge-distance sweep: "
          f"{[(round(f,2), c) for f, c in counts]}")
    print(f"  -> {len(C)} attractors (plateau of {plat} steps, chose d={d:.3f})\n", flush=True)

    print("RUNTIME: scoring test actions ...", flush=True)
    acts = np.stack([tb.random_push(rng, n, thr) for _ in range(args.n_test)])
    marg = np.array([margin_of(a) for a in acts]); near = marg < args.near
    S = []
    for a in acts:
        P = probes("shared-action", a, thr, args.eps, args.n_perturb, rng)
        Ep, _ = settle_latents(model, P, mu, sd, amu, asd, s0, n, st)
        lab = np.argmin(((Ep[:, None] - C[None]) ** 2).sum(-1), axis=1)
        p = np.array([(lab == j).mean() for j in range(len(C))]); p = p[p > 0]
        S.append(float(-(p * np.log(p)).sum()))
    S = np.array(S)
    m = metrics(S, near)
    pred = S > 1e-9
    tp = int((pred & near).sum()); fp = int((pred & ~near).sum()); fn = int((~pred & near).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n{args.n_test} actions, {int(near.sum())} within {args.near:.0%} of the boundary")
    print(f"  AUC (proximity)          {m['auc']:.3f}")
    print(f"  S>0 rule, NO calibration : precision {pr:.3f}  recall {rc:.3f}  "
          f"F1 {2*pr*rc/(pr+rc) if pr+rc else 0:.3f}  flags {100*pred.mean():.0f}%")
    print(f"  best-F1 (cheating, for reference): {m['f1']:.3f}")
    json.dump({"n_attractors": len(C), "merge_d": float(d), "auc": m["auc"],
               "precision": pr, "recall": rc, "sweep": counts},
              open(OUT / "pipeline.json", "w"), indent=1, default=float)
    print(f"\n-> {OUT/'pipeline.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
