"""
Phase 1b — does the piecewise-linear head actually help, or is that just an assumption?

PLAN.md's design commitment #2 says a smooth model "provably cannot represent the
discontinuity". Phase 1b turns that from an assertion into a measurement by running the SAME
training, SAME data (including off-attractor sequences), SAME Benettin/QR spectrum code on:

    shplrnn   piecewise-linear, analytic piecewise-constant Jacobian
    smooth    tanh residual MLP, smooth Jacobian via autograd
    gru       gated recurrence, smooth Jacobian via autograd

Read the result carefully -- Lorenz is a SMOOTH system with no guard surface, so this phase
CANNOT support a claim about discontinuities. What it can establish is the fair baseline:
whether the piecewise-linear head has an intrinsic advantage on smooth dynamics. The expected
and honest outcome is roughly parity. The architecture argument only becomes testable in
Phase 2, where the bouncing ball has a real impact surface; this phase exists so that a Phase-2
gap is attributable to the guard rather than to the architectures differing everywhere.

Parameter counts are matched as closely as the architectures allow and reported, so a
difference cannot be dismissed as one model simply being bigger.
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

import lorenz                                                        # noqa: E402
from ftle import lyapunov_spectrum_general                           # noqa: E402
from shplrnn import ShPLRNN                                          # noqa: E402
from smooth_baselines import SmoothRNN, GRUCellModel, autograd_jacobian  # noqa: E402
from jacobian import jacobian as analytic_jacobian                   # noqa: E402
from gtf import gtf_rollout_loss                                     # noqa: E402
from run_phase1_lorenz import make_data, make_off_attractor_bank, DT # noqa: E402

OUT = ROOT / "results" / "phase1b"
OUT.mkdir(parents=True, exist_ok=True)


def _n_params(kind, H):
    torch.manual_seed(0)
    if kind == "shplrnn":
        m = ShPLRNN(d=3, H=H, obs_dim=3)
    elif kind == "smooth":
        m = SmoothRNN(d=3, H=H, hidden_layers=1)
    else:
        m = GRUCellModel(d=3, H=H)
    return sum(p.numel() for p in m.parameters())


def match_H(kind, target, lo=2, hi=512):
    """Pick H so this architecture lands as close as possible to `target` parameters.

    Without this the comparison is worthless in both directions: at H=128 the raw counts are
    902 (shPLRNN) / 17411 (smooth) / 51459 (GRU), so a shPLRNN win invites "it was starved"
    and a shPLRNN loss invites "the baseline was 57x bigger". Note the smooth baseline uses
    ONE hidden layer here, matching shPLRNN's actual depth -- shPLRNN is a single-hidden-layer
    map, so a 2-layer MLP would not be the like-for-like control either."""
    best = min(range(lo, hi), key=lambda h: abs(_n_params(kind, h) - target))
    return best, _n_params(kind, best)


def build(kind, H, seed):
    torch.manual_seed(seed)
    if kind == "shplrnn":
        m = ShPLRNN(d=3, H=H, obs_dim=3).double()
        return m, (lambda s: analytic_jacobian(s, m)), "analytic"
    if kind == "smooth":
        m = SmoothRNN(d=3, H=H, hidden_layers=1).double()
        return m, (lambda s: autograd_jacobian(s, m)), "autograd"
    if kind == "gru":
        m = GRUCellModel(d=3, H=H).double()
        return m, (lambda s: autograd_jacobian(s, m)), "autograd"
    raise ValueError(kind)


def train_one(kind, seed, epochs, H, off_frac, n_data, n_seq, seq_len=30, batch=128,
              lr=3e-3, alpha=0.1, verbose=False):
    data, mu, sd = make_data(n_data, seed=seed)
    n_off_bank = int(round(n_seq * off_frac))
    off_bank = (make_off_attractor_bank(data, mu, sd, n_off_bank, seq_len, seed=seed)
                if n_off_bank else None)
    n_on_windows = len(data) - seq_len - 1
    model, jac_fn, jac_kind = build(kind, H, seed)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_off_b = int(round(batch * off_frac)); n_on_b = batch - n_off_b

    for ep in range(epochs):
        tot = nb = 0
        for _ in range(20):
            st = torch.randint(0, n_on_windows, (n_on_b,)).tolist()
            parts = [torch.stack([data[i:i + seq_len] for i in st])]
            if off_bank is not None and n_off_b:
                parts.append(off_bank[torch.randint(0, len(off_bank), (n_off_b,))])
            seq = torch.cat(parts, 0)
            loss = gtf_rollout_loss(model, seq, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        if verbose and (ep % max(1, epochs // 4) == 0 or ep == epochs - 1):
            print(f"      ep {ep:3d}  loss {tot/nb:.3e}", flush=True)
    return model, jac_fn, jac_kind, n_par, data


def spectrum_of(model, jac_fn, data, T, burn=2000):
    with torch.no_grad():
        s = model.lift(data[0].clone())
        for _ in range(burn):
            s = model(s)
            if not torch.isfinite(s).all():
                return None
    if not torch.isfinite(s).all():
        return None
    spec = lyapunov_spectrum_general(s, lambda x: model(x), jac_fn, T=T, dt=DT)
    return None if not torch.isfinite(spec).all() else spec.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["shplrnn", "smooth", "gru"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--off-frac", type=float, default=0.5)
    ap.add_argument("--n-data", type=int, default=60000)
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--eval-T", type=int, default=20000)
    ap.add_argument("--out", default="phase1b.json")
    args = ap.parse_args()
    t0 = time.time()

    truth = json.load(open(ROOT / "results" / "phase1" / "stage1_truth.json"))["spectrum"]
    print(f"ground truth: {truth}\n")

    target = _n_params("shplrnn", args.H)
    H_for = {k: match_H(k, target) for k in args.kinds}
    print("parameter matching (target = shPLRNN @ H=%d):" % args.H)
    for k, (h, n) in H_for.items():
        print(f"   {k:<10} H={h:<4} params={n}")
    print()

    rows = []
    for kind in args.kinds:
        print(f"=== {kind} (H={H_for[kind][0]}) ===", flush=True)
        for seed in args.seeds:
            model, jac_fn, jac_kind, n_par, data = train_one(
                kind, seed, args.epochs, H_for[kind][0], args.off_frac, args.n_data,
                args.n_seq, verbose=(seed == args.seeds[0]))
            spec = spectrum_of(model, jac_fn, data, args.eval_T)
            if spec is None:
                print(f"  seed {seed}: DIVERGED"); rows.append(
                    {"kind": kind, "seed": seed, "ok": False, "n_params": n_par}); continue
            err = [abs(spec[i] - truth[i]) for i in range(3)]
            scale = abs(truth[0])
            rel = [err[0] / scale, err[1] / scale, err[2] / abs(truth[2])]
            rows.append({"kind": kind, "seed": seed, "ok": True, "n_params": n_par,
                         "jacobian": jac_kind, "spectrum": [round(v, 4) for v in spec],
                         "rel_err": [round(r, 4) for r in rel],
                         "within_5pct_all3": bool(all(r < 0.05 for r in rel)),
                         "sum": round(sum(spec), 4)})
            print(f"  seed {seed}: {[round(v,4) for v in spec]}  rel err "
                  f"{[round(r,3) for r in rel]}  all3<5% {rows[-1]['within_5pct_all3']}",
                  flush=True)
        ok = [r for r in rows if r["kind"] == kind and r["ok"]]
        if ok:
            S = np.array([r["spectrum"] for r in ok])
            print(f"  mean {np.round(S.mean(0),4).tolist()}  std {np.round(S.std(0),4).tolist()}"
                  f"  params {ok[0]['n_params']}  jac {ok[0]['jacobian']}\n")

    print("=" * 76)
    print(f"{'model':<10}{'params':>9}{'jacobian':>11}{'lam1':>10}{'lam2':>10}{'lam3':>11}{'lam3 err':>10}")
    print("-" * 76)
    print(f"{'TRUTH':<10}{'':>9}{'exact':>11}{truth[0]:>10.4f}{truth[1]:>10.4f}{truth[2]:>11.4f}{'':>10}")
    for kind in args.kinds:
        ok = [r for r in rows if r["kind"] == kind and r["ok"]]
        if not ok:
            print(f"{kind:<10}  all seeds diverged"); continue
        S = np.array([r["spectrum"] for r in ok])
        print(f"{kind:<10}{ok[0]['n_params']:>9}{ok[0]['jacobian']:>11}"
              f"{S[:,0].mean():>10.4f}{S[:,1].mean():>10.4f}{S[:,2].mean():>11.4f}"
              f"{abs(S[:,2].mean()-truth[2])/abs(truth[2])*100:>9.2f}%")
    print("=" * 76)
    print("Lorenz is SMOOTH -- rough parity here is the expected, honest result and does NOT")
    print("refute the piecewise-linear commitment. Phase 2's guard surface is where the")
    print("architectures should diverge, and this baseline is what makes that gap attributable.")
    json.dump({"truth": truth, "args": vars(args), "rows": rows},
              open(OUT / args.out, "w"), indent=1)
    print(f"\n-> {OUT / args.out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
