"""
Phase 2 — bouncing ball on a driven table. The first system in this repo with a real GUARD.

Same two-stage structure as Phase 1, for the same reason: if Stage 1 fails the instrument is
broken, and only if Stage 1 passes does a Stage 2 failure say anything about the model.

    Stage 1  establish and VALIDATE the true spectrum of the hybrid system
    Stage 2  train an shPLRNN on it and compare its spectrum to that truth

Two things differ from Phase 1 and drive the whole design:

1. NO JACOBIAN FOR THE TRUE SYSTEM. At the guard the correct tangent map needs the saltation
   matrix. Ground truth therefore comes from finite differences of the flow map only
   (`bouncing_ball.true_spectrum`), an estimator validated against the analytic-Jacobian
   spectrum on smooth Lorenz, where the two agree to machine precision.

2. OBSERVATION IS 4-D, THE SYSTEM IS 3-D. phi is circular, so it is observed as
   (cos phi, sin phi); see `bouncing_ball.to_obs` for why raw phi is unusable. The observed
   states therefore lie on a 3-manifold in R^4, so the learned model always has at least one
   spurious exponent transverse to it. The spectral-gap check is mandatory here, not optional
   -- Phase 1's d=20 run showed that without it a meaningless number gets reported as
   lambda_3 and nobody notices.
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

import bouncing_ball as bb                                        # noqa: E402
from ftle import lyapunov_spectrum_general                        # noqa: E402
from jacobian import jacobian as analytic_jacobian                # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from gtf import gtf_rollout_loss, gtf_rollout_loss_latent         # noqa: E402

OUT = ROOT / "results" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
DT, OMEGA, E = bb.DT_DEFAULT, bb.OMEGA_DEFAULT, bb.E_DEFAULT
OBS_DIM = 4


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------

def make_data(n=60_000, seed=0):
    """Trajectory in observation coordinates, standardised, plus per-transition impact flags.

    `impacts[i]` marks the transition i -> i+1 as containing an impact, matching
    bb.simulate's convention, so it indexes transitions rather than states."""
    raw, imp = bb.simulate(n, dt=DT, omega=OMEGA, e=E, seed=seed, return_impacts=True)
    raw = raw.numpy()
    assert not bb.simulate.collapsed, "inelastic collapse while generating training data"
    obs = torch.from_numpy(bb.to_obs(raw))
    mu, sd = obs.mean(0), obs.std(0)
    return ((obs - mu) / sd).to(torch.float64), mu, sd, raw, imp


def make_off_attractor_bank(mu, sd, n_off, seq_len, off_eps=0.3, seed=0):
    """Sequences launched OFF the attractor, then rolled forward with the TRUE dynamics.

    Phase 1's single largest result: on-attractor data contains no evidence about contraction
    TRANSVERSE to the attractor, so the most negative exponent cannot be recovered from it
    (lambda_3 error 23.7% -> 0.27% once off-attractor sequences were added, with no cost to
    lambda_1 in a 5-seed control). The same argument applies here.

    Perturbations must not put the ball THROUGH the table: a state with gap < 0 is not merely
    off-attractor, it is outside the system's domain, and `step` would resolve it as an
    immediate spurious impact. Such draws are pushed back above the guard."""
    rng = np.random.default_rng(seed + 9999)
    base = bb.simulate(max(n_off, 1) * 2, dt=DT, omega=OMEGA, e=E, seed=seed + 5).numpy()
    seqs, flag_list = [], []
    for _ in range(n_off):
        s = base[rng.integers(len(base))].copy()
        s[0] += rng.normal(0, off_eps * 3.0)          # height, scaled to its own spread
        s[1] += rng.normal(0, off_eps * 3.0)          # velocity
        s[2] = rng.uniform(0, bb.TWO_PI)              # any table phase
        g = bb.gap(s, OMEGA)
        if g < 0:
            s[0] += -g + 1e-3                         # lift back above the guard
        traj = np.empty((seq_len, 3))
        flags = np.zeros(seq_len, dtype=bool)
        for k in range(seq_len):
            traj[k] = s
            prev_v = s[1]
            s = bb.step(s, DT, OMEGA, E)
            flags[k] = (s[1] - prev_v) > bb.G * DT * 1.5
        seqs.append(bb.to_obs(traj)); flag_list.append(flags)
    bank = torch.from_numpy(np.stack(seqs)).to(torch.float64)
    return (bank - mu) / sd, torch.from_numpy(np.stack(flag_list))


# ---------------------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------------------

def train(seed, epochs, H, d, off_frac, n_data, n_seq, seq_len=30, batch=128, lr=3e-3,
          alpha=0.1, impact_weight=0.0, verbose=False):
    data, mu, sd, raw, imp = make_data(n_data, seed=seed)
    n_off = int(round(n_seq * off_frac))
    bank, bank_imp = (make_off_attractor_bank(mu, sd, n_off, seq_len, seed=seed)
                      if n_off else (None, None))
    n_win = len(data) - seq_len - 1

    torch.manual_seed(seed)
    model = ShPLRNN(d=d, H=H, obs_dim=OBS_DIM).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = gtf_rollout_loss if d == OBS_DIM else gtf_rollout_loss_latent
    n_off_b = int(round(batch * off_frac)); n_on_b = batch - n_off_b

    for ep in range(epochs):
        tot = nb = 0
        for _ in range(20):
            st = torch.randint(0, n_win, (n_on_b,)).tolist()
            parts = [torch.stack([data[i:i + seq_len] for i in st])]
            fparts = [torch.stack([imp[i:i + seq_len - 1] for i in st])]
            if bank is not None and n_off_b:
                idx = torch.randint(0, len(bank), (n_off_b,))
                parts.append(bank[idx]); fparts.append(bank_imp[idx][:, :seq_len - 1])
            w = None
            if impact_weight > 0:
                flags = torch.cat(fparts, 0).to(torch.float64)
                w = 1.0 + impact_weight * flags
                w = w / w.mean()          # keep the loss scale (and so the effective LR) fixed
            loss = loss_fn(model, torch.cat(parts, 0), alpha, step_weights=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        if verbose and (ep % max(1, epochs // 4) == 0 or ep == epochs - 1):
            print(f"      ep {ep:3d}  loss {tot/nb:.3e}", flush=True)
    return model, data


def learned_spectrum(model, data, T, burn=2000):
    """Benettin/QR on the learned model. Its Jacobian IS exact (piecewise constant), so unlike
    the true system no saltation question arises -- the model has no guard, only hyperplanes."""
    with torch.no_grad():
        s = model.lift(data[0].clone())
        for _ in range(burn):
            s = model(s)
            if not torch.isfinite(s).all():
                return None
    spec = lyapunov_spectrum_general(s, lambda x: model(x),
                                     lambda x: analytic_jacobian(x, model), T=T, dt=DT)
    return None if not torch.isfinite(spec).all() else spec.tolist()


def evaluate(spec_full, truth):
    """Compare the top 3 exponents, but only after checking a spectral gap separates them."""
    top3 = spec_full[:3]
    gap = (spec_full[2] - spec_full[3]) if len(spec_full) > 3 else float("inf")
    scale = abs(truth[0])
    err = [abs(top3[i] - truth[i]) for i in range(3)]
    rel = [err[0] / scale, err[1] / scale, err[2] / abs(truth[2])]
    return {
        "spectrum_top3": [round(v, 4) for v in top3],
        "spectrum_full": [round(v, 4) for v in spec_full],
        "abs_err": [round(v, 4) for v in err],
        "rel_err": [round(v, 4) for v in rel],
        "err_note": "lambda_2 error is |absolute| / |lambda_max|, not relative to ~0",
        "spectral_gap_after_3rd": None if gap == float("inf") else round(gap, 4),
        "gap_ok": bool(gap > 0.5 * abs(truth[2])),
        "lambda_max_within_10pct": bool(rel[0] < 0.10),
        "sum": round(sum(top3), 4),
    }


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["truth", "model", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--d", type=int, default=OBS_DIM)
    ap.add_argument("--off-frac", type=float, default=0.5)
    ap.add_argument("--n-data", type=int, default=60000)
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--impact-weight", type=float, default=0.0,
                    help="extra weight on transitions containing an impact; 0 = uniform")
    ap.add_argument("--eval-T", type=int, default=20000)
    ap.add_argument("--truth-T", type=int, default=20000)
    ap.add_argument("--tag", default="phase2")
    args = ap.parse_args()
    t0 = time.time()
    assert args.d >= OBS_DIM, "latent dim must be at least the observation dim"

    truth_path = OUT / "stage1_truth.json"
    if args.stage in ("truth", "both") or not truth_path.exists():
        print("=== STAGE 1: ground truth (finite differences, no Jacobian) ===", flush=True)
        spec, info = bb.true_spectrum(T=args.truth_T, dt=DT, omega=OMEGA, e=E)
        lam_ref, lam_sd, _ = bb.true_lambda_max(n_seeds=8, n_steps=40000, dt=DT,
                                                omega=OMEGA, e=E)
        ok, rep = bb.validate_spectrum(spec, lam_max_ref=lam_ref)
        print(f"  spectrum        {[round(v,4) for v in spec]}")
        print(f"  seeds accepted  {info['n_accepted']}/{len(info['per_seed'])} "
              f"(rejected on |lambda_2| >= 0.01)")
        print(f"  lambda_2 ~ 0    {rep['zero_exponent']:+.4f}   -> {rep['zero_ok']}")
        print(f"  lambda_max vs independent two-particle "
              f"{rep['lambda_max']:.4f} vs {lam_ref:.4f}+-{lam_sd:.4f} "
              f"-> {rep['lambda_max_agrees']}")
        print(f"  STAGE 1 {'PASSED' if ok else 'FAILED'}\n", flush=True)
        json.dump({"spectrum": spec, "validation": rep, "info": info,
                   "two_particle_lambda_max": lam_ref, "two_particle_std": lam_sd,
                   "regime": {"omega": OMEGA, "e": E, "dt": DT, "Gamma": OMEGA ** 2}},
                  open(truth_path, "w"), indent=1)
        assert ok, "Stage 1 failed -- do not interpret any Stage 2 number until it passes"

    if args.stage == "truth":
        print(f"-> {truth_path}   ({time.time()-t0:.0f}s)")
        return

    truth = json.load(open(truth_path))["spectrum"]
    print(f"=== STAGE 2: shPLRNN (d={args.d}, H={args.H}, off_frac={args.off_frac}, "
          f"impact_weight={args.impact_weight}) ===")
    print(f"ground truth: {[round(v,4) for v in truth]}\n", flush=True)

    rows = []
    for seed in args.seeds:
        model, data = train(seed, args.epochs, args.H, args.d, args.off_frac,
                            args.n_data, args.n_seq, alpha=args.alpha,
                            impact_weight=args.impact_weight,
                            verbose=(seed == args.seeds[0]))
        spec = learned_spectrum(model, data, args.eval_T)
        if spec is None:
            print(f"  seed {seed}: DIVERGED", flush=True)
            rows.append({"seed": seed, "ok": False}); continue
        r = evaluate(spec, truth); r.update(seed=seed, ok=True)
        rows.append(r)
        print(f"  seed {seed}: top3 {r['spectrum_top3']}  rel err "
              f"{[round(v,3) for v in r['rel_err']]}  gap {r['spectral_gap_after_3rd']} "
              f"(ok={r['gap_ok']})", flush=True)

    ok_rows = [r for r in rows if r["ok"]]
    print("\n" + "=" * 68)
    print(f"{'':<10}{'lam1':>10}{'lam2':>10}{'lam3':>11}")
    print(f"{'TRUTH':<10}{truth[0]:>10.4f}{truth[1]:>10.4f}{truth[2]:>11.4f}")
    if ok_rows:
        S = np.array([r["spectrum_top3"] for r in ok_rows])
        print(f"{'shPLRNN':<10}{S[:,0].mean():>10.4f}{S[:,1].mean():>10.4f}{S[:,2].mean():>11.4f}")
        print(f"{'std':<10}{S[:,0].std():>10.4f}{S[:,1].std():>10.4f}{S[:,2].std():>11.4f}")
        n_gap = sum(r["gap_ok"] for r in ok_rows)
        print(f"\nspectral gap clean in {n_gap}/{len(ok_rows)} seeds "
              f"-- without a gap the 'top 3' are not the system's exponents")
    print("=" * 68)
    json.dump({"truth": truth, "args": vars(args), "rows": rows},
              open(OUT / f"stage2_{args.tag}.json", "w"), indent=1)
    print(f"-> {OUT / f'stage2_{args.tag}.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
