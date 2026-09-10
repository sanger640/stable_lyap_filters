"""
The two tests Phase 2 exists for.

TEST A — HYPERPLANE ALIGNMENT. An shPLRNN's nonlinearity is relu(W2 s + h2): each row of W2
is a hyperplane, and the model is linear between them. The bouncing ball's guard is
x - A sin(phi) = 0, which in observation coordinates (x, v, cos, sin) is EXACTLY the
hyperplane with normal (1, 0, 0, -A). So "do piecewise-linear models represent
discontinuities?" becomes a direct measurement against a known target.

    The control is the whole test. With H=128 hyperplanes in 4-d, the best alignment to any
    fixed direction is high BY CHANCE. An untrained model at the same init is therefore
    measured too, and only the trained-minus-untrained gap means anything.

TEST B — DIVERGENCE VS HORIZON. Smooth chaos separates as d_T = d_0 exp(lambda T), so
log d_T is LINEAR in T and lambda(T) converges to a plateau. Across a guard, two nearby
trajectories land on opposite sides, separation jumps to a macroscopic value set by the
discontinuity, and then SATURATES -- so log d_T has a kink and lambda(T) decays like 1/T
instead of converging.

    Run on the TRUE system, where we know exactly when impacts happen, so the two populations
    can be separated by ground truth rather than by the signal being tested.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "geometry", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))

import bouncing_ball as bb                                        # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from run_phase2_ball import DT, E, OBS_DIM, OMEGA, make_data, train   # noqa: E402

OUT = ROOT / "results" / "phase2"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# =======================================================================================
# TEST A
# =======================================================================================

def hyperplanes_in_obs_coords(model, mu, sd):
    """Rows of W2 are hyperplanes in NORMALISED latent space; map them back to raw
    observation coordinates so they can be compared with the known guard."""
    W2 = model.W2.detach().numpy()[:, :OBS_DIM]        # (H, 4); latent==obs for d==4
    h2 = model.h2.detach().numpy()
    mu_, sd_ = mu.numpy()[:OBS_DIM], sd.numpy()[:OBS_DIM]
    n = W2 / sd_                                       # normal in raw obs coords
    b = h2 - (W2 * (mu_ / sd_)).sum(axis=1)            # matching offset
    return n, b


def alignment_scores(model, mu, sd, obs_samples):
    """Two metrics per hyperplane: direction alignment with the guard normal, and how well its
    signed distance tracks the true gap over states the system actually visits."""
    n, b = hyperplanes_in_obs_coords(model, mu, sd)
    g = bb.guard_normal_obs()
    norms = np.linalg.norm(n, axis=1) + 1e-300
    cos = np.abs(n @ g) / norms                        # |cos| with the guard normal

    signed = (obs_samples @ n.T + b) / norms           # (N, H) distance to each hyperplane
    true_gap = obs_samples @ np.array([1.0, 0.0, 0.0, -bb.A])
    tg = (true_gap - true_gap.mean()) / (true_gap.std() + 1e-300)
    sg = (signed - signed.mean(0)) / (signed.std(0) + 1e-300)
    corr = np.abs((sg * tg[:, None]).mean(0))          # |Pearson| with the true gap
    return cos, corr


def gate_flip_analysis(model, data, imp):
    """Do the model's ReLU gates FLIP at impacts?

    This replaces a direction-alignment metric that turned out to be measuring nothing. That
    version compared each hyperplane's normal to the guard normal (1,0,0,-A) and its signed
    distance to the true gap. Both saturated: an UNTRAINED model scored |cos|=0.97 and
    |corr|=0.9985, and a meaningless random direction scored 0.9777. Two reasons -- with H=128
    hyperplanes in 4-d some row is close to any fixed direction by chance, and the gap
    x - A sin(phi) is dominated by x (range ~30) over sin(phi) (range 1), so ANY hyperplane
    with an x-component correlates ~1 with it. It measured a scale artefact.

    A gate flip is the model actually switching linear region -- the only mechanism it has for
    representing a discontinuity. If it uses that mechanism for the impact, flips must
    concentrate on impact transitions. This is scale-free and cannot be satisfied by chance:
    an untrained model's gates flip wherever its random hyperplanes happen to fall."""
    with torch.no_grad():
        z = data if data.shape[-1] == model.d else model.lift(data)
        gates = (z @ model.W2.T + model.h2) > 0            # (N, H)
        flips = (gates[1:] != gates[:-1]).sum(dim=1).double().numpy()
    m = imp[:len(flips)].numpy().astype(bool)
    if m.sum() < 5 or (~m).sum() < 5:
        return None
    return {"flips_at_impact": float(flips[m].mean()),
            "flips_elsewhere": float(flips[~m].mean()),
            "ratio": float(flips[m].mean() / (flips[~m].mean() + 1e-12)),
            "n_impact": int(m.sum()), "n_other": int((~m).sum())}


def test_a(args):
    print("=== TEST A: hyperplane alignment with the guard ===", flush=True)
    data, mu, sd, raw, imp = make_data(args.n_data, seed=0)
    obs = bb.to_obs(bb.simulate(4000, dt=DT, omega=OMEGA, e=E, seed=11).numpy())

    print(f"  training (epochs={args.epochs}, alpha={args.alpha}) ...", flush=True)
    model, _ = train(0, args.epochs, args.H, OBS_DIM, args.off_frac, args.n_data,
                     args.n_seq, alpha=args.alpha, impact_weight=args.impact_weight)
    cos_t, corr_t = alignment_scores(model, mu, sd, obs)

    torch.manual_seed(0)
    untrained = ShPLRNN(d=OBS_DIM, H=args.H, obs_dim=OBS_DIM).double()
    cos_u, corr_u = alignment_scores(untrained, mu, sd, obs)

    # second control: a direction with no dynamical meaning
    rng = np.random.default_rng(0)
    rand_dir = rng.normal(size=4); rand_dir /= np.linalg.norm(rand_dir)
    n_t, _ = hyperplanes_in_obs_coords(model, mu, sd)
    cos_rand = np.abs(n_t @ rand_dir) / (np.linalg.norm(n_t, axis=1) + 1e-300)

    res = {
        "trained_max_cos": float(cos_t.max()), "untrained_max_cos": float(cos_u.max()),
        "trained_top5_cos": np.sort(cos_t)[-5:][::-1].round(4).tolist(),
        "untrained_top5_cos": np.sort(cos_u)[-5:][::-1].round(4).tolist(),
        "trained_max_cos_random_direction": float(cos_rand.max()),
        "trained_max_corr_with_gap": float(corr_t.max()),
        "untrained_max_corr_with_gap": float(corr_u.max()),
        "trained_top5_corr": np.sort(corr_t)[-5:][::-1].round(4).tolist(),
        "n_hyperplanes": int(len(cos_t)),
    }
    print(f"  |cos| with guard normal   trained {res['trained_max_cos']:.4f}   "
          f"untrained {res['untrained_max_cos']:.4f}   "
          f"(vs a random direction: {res['trained_max_cos_random_direction']:.4f})")
    print(f"  |corr| with true gap      trained {res['trained_max_corr_with_gap']:.4f}   "
          f"untrained {res['untrained_max_corr_with_gap']:.4f}")
    res["alignment_metric_is_uninformative"] = bool(
        res["untrained_max_cos"] > 0.9 or res["untrained_max_corr_with_gap"] > 0.9)

    gt = gate_flip_analysis(model, data, imp)
    gu = gate_flip_analysis(untrained, data, imp)
    res["gate_flips_trained"], res["gate_flips_untrained"] = gt, gu
    print("  --- gate flips per transition (the informative metric) ---")
    for name, g in (("trained", gt), ("untrained", gu)):
        if g:
            print(f"    {name:>9}: at impact {g['flips_at_impact']:6.2f}   "
                  f"elsewhere {g['flips_elsewhere']:6.2f}   ratio {g['ratio']:5.2f}x")
    verdict = bool(gt and gu and gt["ratio"] > 1.5 * max(gu["ratio"], 1.0))
    res["trained_beats_control"] = verdict
    print(f"  -> trained concentrates gate flips at the guard: {verdict}\n", flush=True)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(["impact", "elsewhere"], [gt["flips_at_impact"], gt["flips_elsewhere"]],
            color=["#c0392b", "#95a5a6"], alpha=0.85, label="trained")
    ax2.plot(["impact", "elsewhere"], [gu["flips_at_impact"], gu["flips_elsewhere"]],
             "ko--", label="untrained (control)")
    ax2.set_ylabel("ReLU gate flips per transition")
    ax2.set_title("Test A — does the model switch linear region at the guard?")
    ax2.legend(fontsize=8); fig2.tight_layout()
    fig2.savefig(FIG / "test_a_gate_flips.png", dpi=130); plt.close(fig2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (t, u, name) in zip(ax, [(cos_t, cos_u, "|cos| with guard normal"),
                                    (corr_t, corr_u, "|corr| with true gap")]):
        a.hist(u, bins=30, alpha=0.6, label="untrained (control)", color="#95a5a6")
        a.hist(t, bins=30, alpha=0.6, label="trained", color="#c0392b")
        a.set_xlabel(name); a.set_ylabel("hyperplanes"); a.legend(fontsize=8)
    fig.suptitle("Test A — do learned ReLU boundaries land on the guard?")
    fig.tight_layout(); fig.savefig(FIG / "test_a_alignment.png", dpi=130)
    plt.close(fig)
    return res


# =======================================================================================
# TEST B
# =======================================================================================

def test_b(args):
    """Split by STRADDLING, not by 'an impact happened'.

    First attempt grouped on whether the reference trajectory hit the guard, and found NO
    difference -- both groups converged to ~0.20. That null is correct and worth keeping: for
    INFINITESIMAL perturbations the tangent map across a guard is the saltation matrix, which
    is finite, so the exponent converges normally. It is what Stage 1 measures.

    The divergence only exists for FINITE perturbations whose two members land on OPPOSITE
    sides of the guard -- one bounces, the other does not. Separation then jumps to a value set
    by the discontinuity rather than by eps. That is the monitor's actual regime (Jenga
    perturbs with eps=0.005, not 1e-9), so straddling is the effect that could transfer."""
    print("=== TEST B: divergence vs horizon on the TRUE system ===", flush=True)
    Tmax, N = args.horizon, args.n_pairs
    rng = np.random.default_rng(0)
    base = bb.simulate(N * 3, dt=DT, omega=OMEGA, e=E, seed=7).numpy()
    out = {}

    for eps in args.eps:
        D = np.zeros((N, Tmax + 1))
        straddle = np.zeros((N, Tmax), dtype=bool)
        for i in range(N):
            s = base[rng.integers(len(base))].copy()
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            s2 = s + eps * v
            D[i, 0] = eps
            c1 = c2 = 0
            for t in range(1, Tmax + 1):
                p1, p2 = s[1], s2[1]
                s = bb.step(s, DT, OMEGA, E); s2 = bb.step(s2, DT, OMEGA, E)
                c1 += (s[1] - p1) > bb.G * DT * 1.5
                c2 += (s2[1] - p2) > bb.G * DT * 1.5
                straddle[i, t - 1] = (c1 != c2)      # the pair disagrees about bouncing
                d = s2 - s
                d[2] = (d[2] + np.pi) % bb.TWO_PI - np.pi
                D[i, t] = np.linalg.norm(d)
        t_ax = np.arange(1, Tmax + 1)
        lam = np.log(D[:, 1:] / eps) / (t_ax * DT)
        final = straddle[:, -1]
        grp = {"straddle_fraction": float(final.mean())}
        for name, m in (("straddled", final), ("clean", ~final)):
            if m.sum() < 5:
                continue
            grp[name] = {
                "n": int(m.sum()),
                "median_log10_ratio": np.log10(np.median(D[m, 1:], axis=0) / eps).round(4).tolist(),
                "median_ftle": np.median(lam[m], axis=0).round(4).tolist(),
            }
        out[f"eps_{eps:.0e}"] = grp
        print(f"  eps={eps:.0e}  straddling pairs: {100*final.mean():.1f}%", flush=True)
        for name in ("straddled", "clean"):
            if name not in grp:
                continue
            f = grp[name]["median_ftle"]
            print(f"      {name:>10} (n={grp[name]['n']:3d}): FTLE  T=1 {f[0]:+.3f}   "
                  f"T={Tmax//4} {f[Tmax//4-1]:+.3f}   T={Tmax} {f[-1]:+.3f}", flush=True)

    # ---- figure -------------------------------------------------------------------------
    t_ax = np.arange(1, Tmax + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for eps, ls in zip(args.eps, ["-", "--", ":"]):
        g = out[f"eps_{eps:.0e}"]
        for name, col in (("clean", "#2980b9"), ("straddled", "#c0392b")):
            if name not in g:
                continue
            lbl = f"{name} (eps={eps:.0e})"
            ax[0].plot(t_ax, g[name]["median_log10_ratio"], ls, color=col, lw=1.6, label=lbl)
            ax[1].plot(t_ax, g[name]["median_ftle"], ls, color=col, lw=1.6, label=lbl)
    ax[0].set_xlabel("horizon T (steps)"); ax[0].set_ylabel("median log10(d_T / d_0)")
    ax[0].set_title("separation — straight line = smooth stretching")
    ax[1].axhline(0.1585, color="k", ls="--", lw=0.8, label="true lambda_1")
    ax[1].set_xlabel("horizon T (steps)"); ax[1].set_ylabel("FTLE at horizon T")
    ax[1].set_title("FTLE — plateau = converged, decay = saturated")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle("Test B — does straddling the guard leave a signature?")
    fig.tight_layout(); fig.savefig(FIG / "test_b_horizon.png", dpi=130)
    plt.close(fig)
    return out


# =======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=["a", "b"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--impact-weight", type=float, default=20.0)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--off-frac", type=float, default=0.5)
    ap.add_argument("--n-data", type=int, default=60000)
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    args = ap.parse_args()

    res = {"args": vars(args)}
    if "b" in args.tests:
        res["test_b"] = test_b(args)
    if "a" in args.tests:
        res["test_a"] = test_a(args)
    json.dump(res, open(OUT / "phase2_tests.json", "w"), indent=1)
    print(f"-> {OUT / 'phase2_tests.json'}\n-> {FIG}")


if __name__ == "__main__":
    main()
