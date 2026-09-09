"""
Phase 1 — Lorenz-63: does the FTLE machinery work at all?

Two-stage validation, deliberately separable:
  Stage 1  spectrum from the TRUE equations, via the same Benettin/QR code.
           Wrong here => the FTLE code is broken. Everything else is unfalsifiable.
  Stage 2  spectrum from the LEARNED shPLRNN.
           Right in stage 1 but wrong here => the model's Jacobians are wrong, which is
           exactly the failure the separation loss exists to fix.

Acceptance (stated before running, per PLAN.md §4.3):
  * all THREE exponents within 5% of ground truth, over 5 seeds
  * the NEGATIVE exponent recovered, not just lambda_max -- recovering only lambda_max is the
    standard signature of a model whose trajectories look fine and whose derivatives do not
  * the ablation shows whether the separation loss measurably helps (either answer is a result)

Note on coordinates: training runs on standardised states. Lyapunov exponents are invariant
under an invertible linear change of coordinates (the Jacobian products are related by a
similarity transform), so exponents measured in normalised space transfer directly.
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

import lorenz                                                        # noqa: E402
from ftle import lyapunov_spectrum_general, spectrum_convergence     # noqa: E402
from shplrnn import ShPLRNN                                          # noqa: E402
from jacobian import jacobian                                        # noqa: E402
from gtf import gtf_rollout_loss, gtf_rollout_loss_latent            # noqa: E402
from pairs import mine_pairs                                         # noqa: E402
from neighborhood_loss import separation_loss                        # noqa: E402

OUT = ROOT / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.01


def make_data(n=60_000, dt=DT, seed=0):
    traj = lorenz.simulate(n, dt=dt, seed=seed)
    mu, sd = traj.mean(0), traj.std(0)
    return ((traj - mu) / sd).to(torch.float64), mu, sd


def make_off_attractor_bank(traj_norm, mu, sd, n_off, seq_len, dt=DT, off_eps=0.3, seed=0):
    """Pre-generate OFF-attractor transients only (an RK4 loop each, so this is the expensive
    part). On-attractor windows are NOT pre-generated -- they are resampled fresh every batch
    from the full trajectory.

    An earlier version baked BOTH into one fixed bank. That cut effective on-attractor
    diversity ~7x and dropped lambda_1 from 0.888 to 0.826 at otherwise identical settings,
    which looked like an off-attractor tradeoff and was not one."""
    g = torch.Generator().manual_seed(seed + 99)
    hi = len(traj_norm) - seq_len - 1
    idxs = torch.randint(0, hi, (n_off,), generator=g).tolist()
    v = torch.randn(n_off, 3, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    out = torch.empty(n_off, seq_len, 3, dtype=torch.float64)
    for k, idx in enumerate(idxs):
        raw = (traj_norm[idx] + off_eps * v[k]) * sd + mu     # kick off the attractor
        for t in range(seq_len):                              # record the transient return
            out[k, t] = raw
            raw = lorenz.rk4_step(raw, dt)
    return (out - mu) / sd


def _legacy_make_sequences(traj_norm, mu, sd, n_seq, seq_len, dt=DT, off_frac=0.0, off_eps=0.3, seed=0):
    """Training sequences, optionally including OFF-ATTRACTOR transients.

    Why this exists. All on-attractor data is, by construction, evidence about motion ALONG
    the attractor -- which is what lambda_1 and lambda_2 govern. lambda_3 governs how fast a
    deviation TRANSVERSE to the attractor collapses, and that motion decays to nothing during
    the burn-in we deliberately discard. So an on-attractor dataset contains essentially no
    evidence about lambda_3, and whatever the model reports for it is inductive bias rather
    than a fit to data.

    (This also explains why mined near-neighbour pairs cannot fix it: both members of any
    mined pair come from the data, hence both lie on the attractor, so their separation is
    tangent to it. No amount of such pairs probes the transverse direction.)

    Fix: deliberately start states OFF the attractor and record the return. In simulation the
    initial condition is ours to choose, so this evidence is free.
    """
    g = torch.Generator().manual_seed(seed)
    n_off = int(round(n_seq * off_frac))
    n_on = n_seq - n_off
    seqs = []

    hi = len(traj_norm) - seq_len - 1
    for idx in torch.randint(0, hi, (n_on,), generator=g).tolist():
        seqs.append(traj_norm[idx:idx + seq_len])

    if n_off:
        base_idx = torch.randint(0, hi, (n_off,), generator=g).tolist()
        v = torch.randn(n_off, 3, generator=g, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True)
        for k, idx in enumerate(base_idx):
            s0_norm = traj_norm[idx] + off_eps * v[k]          # kick off the attractor
            s = s0_norm * sd + mu                              # back to raw coords to simulate
            out = torch.empty(seq_len, 3, dtype=torch.float64)
            for t in range(seq_len):                           # record the TRANSIENT return
                out[t] = s
                s = lorenz.rk4_step(s, dt)
            seqs.append((out - mu) / sd)
    return torch.stack(seqs)


def train(seed, alpha, beta, epochs, H, seq_len, batch, lr, n_data, sep_horizon,
          radius, second_order_eps, off_frac=0.0, off_eps=0.3, n_seq=4000, d=3,
          verbose=False):
    torch.manual_seed(seed)
    data, mu, sd = make_data(n_data, seed=seed)
    n_off_bank = int(round(n_seq * off_frac))
    off_bank = (make_off_attractor_bank(data, mu, sd, n_off_bank, seq_len, off_eps=off_eps,
                                        seed=seed) if n_off_bank else None)
    n_on_windows = len(data) - seq_len - 1
    if verbose:
        print(f"    on-attractor: {n_on_windows} windows resampled per batch | "
              f"off-attractor bank: {n_off_bank} (eps={off_eps}), {off_frac:.0%} of each batch")
    model = ShPLRNN(d=d, H=H, obs_dim=3).double()
    latent_mode = d > 3
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    pairs = None
    if beta > 0:
        pairs = mine_pairs(data, horizon=sep_horizon, radius=radius, seed=seed)
        if verbose:
            print(f"    mined {pairs['n_pairs'] if pairs else 0} pairs (radius={radius})")

    hist = []
    n_off_b = int(round(batch * off_frac))
    n_on_b = batch - n_off_b
    for ep in range(epochs):
        ep_pred = ep_sep = 0.0
        nb = 0
        for _ in range(20):
            st = torch.randint(0, n_on_windows, (n_on_b,)).tolist()
            parts = [torch.stack([data[i:i + seq_len] for i in st])]
            if off_bank is not None and n_off_b:
                parts.append(off_bank[torch.randint(0, len(off_bank), (n_off_b,))])
            seq = torch.cat(parts, 0)
            l_pred = (gtf_rollout_loss_latent(model, seq, alpha) if latent_mode
                      else gtf_rollout_loss(model, seq, alpha))
            loss = l_pred
            l_sep = torch.tensor(0.0)
            if pairs is not None and pairs["n_pairs"] > 0:
                sub = torch.randperm(pairs["n_pairs"])[:512]
                p = {"i": pairs["i"][sub], "j": pairs["j"][sub]}
                l_sep = separation_loss(model, data, p, sep_horizon,
                                        second_order_eps=second_order_eps, seed=seed)
                loss = loss + beta * l_sep
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ep_pred += float(l_pred.detach()); ep_sep += float(l_sep.detach()); nb += 1
        sched.step()
        hist.append({"epoch": ep, "pred": ep_pred / nb, "sep": ep_sep / nb})
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print(f"    ep {ep:3d}  pred {ep_pred/nb:.3e}  sep {ep_sep/nb:.3e}")
    return model, data, hist


def learned_spectrum(model, data, T=50_000, burn=2000):
    """Free-running rollout of the LEARNED model, then Benettin/QR on its own attractor.

    Uses the model's own trajectory, not the data's: we are characterising what the model
    learned. If it settled onto a different attractor, this is what exposes it."""
    with torch.no_grad():
        s = model.lift(data[0].clone())          # observation -> latent (identity when d==3)
        for _ in range(burn):
            s = model(s)
            if not torch.isfinite(s).all():
                return None, "diverged during burn-in"
        spec = lyapunov_spectrum_general(s, lambda x: model(x),
                                         lambda x: jacobian(x, model), T=T, dt=DT)
    return (None, "non-finite spectrum") if not torch.isfinite(spec).all() else (spec, None)


def evaluate(model, data, truth, T):
    spec, err = learned_spectrum(model, data, T=T)
    if spec is None:
        return {"ok": False, "reason": err}
    spec_full = spec.tolist()
    # With a latent larger than the observation there are d exponents, not 3. The top 3 should
    # be the real dynamics and the rest strongly negative (collapse onto the learned
    # 3-manifold). Compare the top 3, but ALSO report the gap after the 3rd -- without a clear
    # gap the "top 3" may be contaminated by spurious latent directions, and the comparison
    # would be meaningless in a way that looks fine.
    spec = spec_full[:3]
    gap = (spec_full[2] - spec_full[3]) if len(spec_full) > 3 else float("inf")
    absd = [abs(spec[i] - truth[i]) for i in range(3)]
    # lambda_2 is ~0 by construction, so a RELATIVE error against it is meaningless (it blows
    # up to hundreds and says nothing). Judge the zero exponent on an absolute scale, taken
    # relative to the size of lambda_max -- that is the only self-consistent choice.
    scale = max(abs(truth[0]), 1e-9)
    rel = [absd[0] / scale, absd[1] / scale, absd[2] / max(abs(truth[2]), 1e-9)]
    ok_sum, rep = lorenz.validate_spectrum(spec, tol_trace=1.0, tol_zero=0.1)
    return {"ok": True, "spectrum": [round(v, 4) for v in spec],
            "n_exponents": len(spec_full),
            "spectral_gap_after_3rd": round(gap, 4) if gap != float("inf") else None,
            "spurious_exponents": [round(v, 3) for v in spec_full[3:8]],
            "rel_err": [round(r, 4) for r in rel], "abs_err": [round(a, 4) for a in absd],
            "err_note": "lambda_2 error is |absolute| / |lambda_max|, not relative to ~0",
            "within_5pct_all3": bool(all(r < 0.05 for r in rel)),
            "lambda_max_within_5pct": bool(rel[0] < 0.05),
            "lambda_zero_ok": bool(rel[1] < 0.05),
            "lambda_neg_within_5pct": bool(rel[2] < 0.05),
            "sum": rep["sum"], "sum_error_vs_trace": rep["sum_error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-data", type=int, default=40_000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--sep-horizon", type=int, default=10)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--second-order-eps", type=float, default=0.0)
    ap.add_argument("--eval-T", type=int, default=50_000)
    ap.add_argument("--truth-T", type=int, default=200_000)
    ap.add_argument("--stage", choices=["truth", "train", "both"], default="both")
    ap.add_argument("--off-frac", type=float, default=0.0,
                    help="fraction of training sequences started OFF the attractor")
    ap.add_argument("--off-eps", type=float, default=0.3,
                    help="off-attractor kick size, in units of the data std")
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--d", type=int, default=3, help="model state dim (rule 4.7: do not raise silently)")
    ap.add_argument("--sweep", choices=["none", "alpha", "beta", "ablation", "offattr"],
                    default="none")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()

    # ---------------- stage 1: is the instrument correct? ----------------
    truth_path = OUT / "stage1_truth.json"
    if truth_path.exists() and args.stage != "truth":
        truth_rec = json.load(open(truth_path))
        print(f"stage 1: reusing {truth_path.name}  {truth_rec['spectrum']}")
    else:
        print(f"stage 1: ground-truth spectrum from the TRUE equations (T={args.truth_T}) ...",
              flush=True)
        spec = lorenz.true_spectrum(T=args.truth_T, dt=DT)
        ok, rep = lorenz.validate_spectrum(spec)
        rep["published"] = list(lorenz.PUBLISHED_SPECTRUM)
        rep["abs_err_vs_published"] = [round(abs(float(spec[i]) - lorenz.PUBLISHED_SPECTRUM[i]), 4)
                                       for i in range(3)]
        rep["reference_free_checks_passed"] = bool(ok)
        json.dump(rep, open(truth_path, "w"), indent=1)
        truth_rec = rep
        print(f"  spectrum {rep['spectrum']}")
        print(f"  sum {rep['sum']} vs trace {rep['expected_sum_trace']} (err {rep['sum_error']})")
        print(f"  zero-exponent err {rep['closest_to_zero']}   checks passed: {ok}")
        if not ok:
            print("\nSTOP: the FTLE code fails on the TRUE system. Fix before training.")
            return
    truth = truth_rec["spectrum"]
    if args.stage == "truth":
        return

    # ---------------- stage 2: is the learned model's geometry right? ----------------
    configs = [{"alpha": args.alpha, "beta": args.beta, "name": "default"}]
    if args.sweep == "alpha":
        configs = [{"alpha": a, "beta": args.beta, "name": f"alpha={a}"}
                   for a in (0.02, 0.05, 0.1, 0.2, 0.4)]
    elif args.sweep == "beta":
        configs = [{"alpha": args.alpha, "beta": b, "name": f"beta={b}"}
                   for b in (0.0, 0.1, 1.0, 10.0)]
    elif args.sweep == "offattr":
        configs = [{"alpha": args.alpha, "beta": args.beta, "off_frac": f,
                    "name": f"off_frac={f}"} for f in (0.0, 0.25, 0.5, 0.75)]
    elif args.sweep == "ablation":
        configs = [{"alpha": args.alpha, "beta": 0.0, "name": "no separation loss"},
                   {"alpha": args.alpha, "beta": args.beta, "name": "with separation loss"}]

    all_rows = []
    for cfg in configs:
        print(f"\n=== {cfg['name']}  (alpha={cfg['alpha']}, beta={cfg['beta']}) ===", flush=True)
        rows = []
        for seed in args.seeds:
            model, data, hist = train(seed, cfg["alpha"], cfg["beta"], args.epochs, args.H,
                                      args.seq_len, args.batch, args.lr, args.n_data,
                                      args.sep_horizon, args.radius, args.second_order_eps,
                                      off_frac=cfg.get("off_frac", args.off_frac),
                                      off_eps=args.off_eps, n_seq=args.n_seq, d=args.d,
                                      verbose=(seed == args.seeds[0]))
            r = evaluate(model, data, truth, args.eval_T)
            r.update(seed=seed, **{k: v for k, v in cfg.items() if k != "name"},
                     off_eps=args.off_eps, d=args.d,
                     final_pred_loss=hist[-1]["pred"], final_sep_loss=hist[-1]["sep"])
            rows.append(r); all_rows.append({**r, "config": cfg["name"]})
            if r["ok"]:
                extra = ("" if r["spectral_gap_after_3rd"] is None
                         else f"   gap after 3rd {r['spectral_gap_after_3rd']:.2f}")
                print(f"  seed {seed}: {r['spectrum']}   rel err {r['rel_err']}   "
                      f"all3<5% {r['within_5pct_all3']}{extra}")
            else:
                print(f"  seed {seed}: FAILED ({r['reason']})")
        ok_rows = [r for r in rows if r["ok"]]
        if ok_rows:
            arr = np.array([r["spectrum"] for r in ok_rows])
            print(f"  mean spectrum {np.round(arr.mean(0), 4).tolist()}  "
                  f"std {np.round(arr.std(0), 4).tolist()}")
            print(f"  all-3-within-5%: {sum(r['within_5pct_all3'] for r in ok_rows)}/{len(ok_rows)}"
                  f"   lambda_max only: {sum(r['lambda_max_within_5pct'] for r in ok_rows)}/{len(ok_rows)}"
                  f"   negative exp: {sum(r['lambda_neg_within_5pct'] for r in ok_rows)}/{len(ok_rows)}")

    out = OUT / f"stage2_{args.tag}.json"
    json.dump({"truth": truth, "args": vars(args), "rows": all_rows}, open(out, "w"), indent=1)
    print(f"\n-> {out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
