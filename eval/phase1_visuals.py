"""
Phase 1 visuals: what the learned model actually produces, not just its exponents.

Trains (or reloads) the accepted configuration -- d=3, beta=0, off_frac=0.5 -- saves the
checkpoint, then generates:

  1. attractor      true Lorenz vs the model's FREE-RUNNING attractor (no data injected)
  2. tracking       both started from the SAME initial condition, per-coordinate vs time
  3. convergence    running spectrum estimate vs horizon T -- must FLATTEN here; at a guard
                    surface (Phase 2) the same plot diverges instead, and learning to read it
                    on a system where we know the answer is the point of this phase
  4. return_map     successive maxima of z, true vs learned -- the diagnostic Pathak et al.
                    use to explain WHY lambda_3 is hard: the apparent curve has a thin
                    transverse thickness, and that thickness is the only orbital evidence of
                    lambda_3
  5. off_ablation   lambda_3 vs off-attractor data fraction (the headline Phase-1 finding)
  6. dim_ladder     d=3 vs d=20 full spectra -- why the d=20 "top 3" is contaminated
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "geometry", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))

import lorenz                                                    # noqa: E402
from ftle import spectrum_convergence                            # noqa: E402
from shplrnn import ShPLRNN                                      # noqa: E402
from jacobian import jacobian                                    # noqa: E402
from run_phase1_lorenz import train, make_data, DT               # noqa: E402

M = ROOT / "media"
CKPT = ROOT / "results" / "phase1" / "model_d3_off50_seed2.pt"
C_TRUE, C_LEARN, C_OK, C_BAD = "#333333", "#8172B3", "#55A868", "#C44E52"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def get_model(seed=2, epochs=300, force=False):
    if CKPT.exists() and not force:
        blob = torch.load(CKPT, weights_only=False)
        m = ShPLRNN(d=3, H=blob["H"], obs_dim=3).double()
        m.load_state_dict(blob["state_dict"])
        print(f"  loaded {CKPT.name}")
        return m, blob["mu"], blob["sd"]
    print(f"  training d=3 off_frac=0.5 seed={seed} ({epochs} epochs) ...", flush=True)
    model, data, _ = train(seed=seed, alpha=0.1, beta=0.0, epochs=epochs, H=128, seq_len=30,
                           batch=128, lr=3e-3, n_data=60000, sep_horizon=10, radius=0.05,
                           second_order_eps=0.0, off_frac=0.5, off_eps=0.3, n_seq=4000, d=3,
                           verbose=True)
    _, mu, sd = make_data(60000, seed=seed)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "H": 128, "mu": mu, "sd": sd,
                "config": "d=3 H=128 beta=0 off_frac=0.5 epochs=300"}, CKPT)
    print(f"  saved {CKPT.name}")
    return model, mu, sd


def rollout(model, s0_norm, steps):
    """FREE-running rollout: no data injected at any point."""
    out, s = [], s0_norm.clone()
    with torch.no_grad():
        for _ in range(steps):
            s = model(s)
            out.append(s.clone())
    return torch.stack(out)


def fig_attractor(model, mu, sd):
    true = lorenz.simulate(20000, dt=DT, seed=7)
    s0 = ((true[0] - mu) / sd)
    learned = rollout(model, s0, 20000) * sd + mu
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, (i, j, lab) in zip(axes, [(0, 2, "x–z"), (0, 1, "x–y"), (1, 2, "y–z")]):
        ax.plot(true[:, i], true[:, j], lw=.25, color=C_TRUE, alpha=.75, label="true Lorenz")
        ax.plot(learned[:, i], learned[:, j], lw=.25, color=C_LEARN, alpha=.75,
                label="learned (free-running)")
        ax.set_title(lab, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Learned model's free-running attractor vs the true one\n"
                 "no data is injected — the model generates this entirely on its own",
                 fontsize=10, y=1.05)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_attractor.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_attractor")


def fig_tracking(model, mu, sd, steps=1200):
    true = lorenz.simulate(steps, dt=DT, seed=11)
    s0 = ((true[0] - mu) / sd)
    learned = rollout(model, s0, steps) * sd + mu
    t = np.arange(steps) * DT
    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        ax.plot(t, true[:, k], color=C_TRUE, lw=1.1, label="true")
        ax.plot(t, learned[:, k], color=C_LEARN, lw=1.1, ls="--", label="learned")
        ax.set_ylabel(name)
    err = (true - learned).norm(dim=-1)
    idx = int(torch.nonzero(err > 1.0)[0]) if (err > 1.0).any() else steps - 1
    for ax in axes:
        ax.axvline(t[idx], color=C_BAD, ls=":", lw=1.2)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[0].set_title(f"Same initial condition, both run freely — tracks for ~{t[idx]:.1f} "
                      f"time units, then separates\n"
                      f"separation is EXPECTED: lambda_1>0 means any error grows like e^(0.9t). "
                      f"Attractor shape is the meaningful test, not pointwise tracking",
                      fontsize=9.5)
    axes[-1].set_xlabel("time")
    fig.tight_layout(); fig.savefig(M / "fig_phase1_tracking.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_tracking")


def fig_convergence(model, mu, sd):
    data, _, _ = make_data(3000, seed=5)
    s = data[0].clone()
    with torch.no_grad():
        for _ in range(2000):
            s = model(s)
    cps = [500, 1000, 2000, 5000, 10000, 20000, 30000]
    learned = spectrum_convergence(s, lambda x: model(x), lambda x: jacobian(x, model),
                                   cps, dt=DT)
    truth = json.load(open(ROOT / "results" / "phase1" / "stage1_truth.json"))["spectrum"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, ax in enumerate(axes):
        ax.plot(cps, [learned[T][k] for T in cps], "o-", color=C_LEARN, lw=1.6, ms=4,
                label="learned")
        ax.axhline(truth[k], color=C_TRUE, ls="--", lw=1.3, label="truth")
        ax.set_xscale("log"); ax.set_xlabel("horizon T (steps)")
        ax.set_title(f"$\\lambda_{k+1}$", fontsize=10)
    axes[0].set_ylabel("exponent estimate"); axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Spectrum estimate vs horizon — it FLATTENS, i.e. converges\n"
                 "at a true guard surface (Phase 2) this same plot diverges instead; "
                 "that contrast is what makes the measurement interpretable",
                 fontsize=9.5, y=1.06)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_convergence.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_convergence")


def _z_maxima(traj):
    z = traj[:, 2].numpy()
    pk = np.where((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
    return z[pk]


def fig_return_map(model, mu, sd):
    true = lorenz.simulate(60000, dt=DT, seed=13)
    s0 = ((true[0] - mu) / sd)
    learned = rollout(model, s0, 60000) * sd + mu
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
    for ax, traj, c, name in [(axes[0], true, C_TRUE, "true Lorenz"),
                              (axes[1], learned, C_LEARN, "learned model")]:
        zm = _z_maxima(traj)
        ax.plot(zm[:-1], zm[1:], ".", ms=1.1, color=c, alpha=.55)
        ax.set_xlabel("$z_n$ (n-th maximum)"); ax.set_title(name, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("$z_{n+1}$")
    fig.suptitle("Lorenz return map — successive maxima of z\n"
                 "Pathak et al. explain lambda_3's difficulty here: this apparent CURVE has a "
                 "thin transverse thickness,\nand that thickness is the only orbital evidence "
                 "of the contracting exponent",
                 fontsize=9.5, y=1.08)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_return_map.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_return_map")


def fig_off_ablation():
    f = ROOT / "results" / "phase1" / "stage2_offattr.json"
    if not f.exists():
        print("  skip fig_phase1_off_ablation"); return
    d = json.load(open(f)); truth = d["truth"]
    rows = [r for r in d["rows"] if r.get("ok")]
    cfgs = sorted({r["config"] for r in rows}, key=lambda c: float(c.split("=")[-1]))
    fr = [float(c.split("=")[-1]) for c in cfgs]
    mu = [np.mean([r["spectrum"][2] for r in rows if r["config"] == c]) for c in cfgs]
    sd_ = [np.std([r["spectrum"][2] for r in rows if r["config"] == c]) for c in cfgs]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    band = .05 * abs(truth[2])
    ax.axhspan(truth[2] - band, truth[2] + band, color=C_OK, alpha=.18, label="±5% accept band")
    ax.axhline(truth[2], color=C_TRUE, ls="--", lw=1.4, label=f"truth {truth[2]:.3f}")
    ax.errorbar([f * 100 for f in fr], mu, yerr=sd_, fmt="o-", color=C_LEARN, lw=1.8,
                ms=6, capsize=3, label=r"learned $\lambda_3$")
    for f_, m in zip(fr, mu):
        ax.annotate(f"{abs(m-truth[2])/abs(truth[2])*100:.0f}%", (f_ * 100, m),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
    ax.axhline(-10.5, color=C_BAD, ls=":", lw=1.3)
    ax.text(2, -10.3, "Pathak et al. reservoir: −10.5 (28% error)", fontsize=7.5, color=C_BAD)
    ax.set_xlabel("% of training sequences started OFF the attractor")
    ax.set_ylabel(r"$\lambda_3$")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("THE Phase-1 finding: off-attractor data fixes the contracting exponent\n"
                 "on-attractor data contains almost no evidence about it", fontsize=9.5)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_off_ablation.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_off_ablation")


def fig_dim_ladder():
    f3 = ROOT / "results" / "phase1" / "stage2_confirm5.json"
    f20 = ROOT / "results" / "phase1" / "stage2_d20.json"
    if not (f3.exists() and f20.exists()):
        print("  skip fig_phase1_dim_ladder"); return
    d3, d20 = json.load(open(f3)), json.load(open(f20))
    truth = d3["truth"]
    r3 = [r for r in d3["rows"] if r.get("ok")][0]
    r20 = [r for r in d20["rows"] if r.get("ok")][0]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, v in enumerate(truth):
        ax.plot([0.6, 1.4], [v, v], color=C_TRUE, lw=2.2,
                label="true Lorenz" if k == 0 else None)
    for k, v in enumerate(r3["spectrum"]):
        ax.plot([1.6, 2.4], [v, v], color=C_OK, lw=2.2, label="d=3" if k == 0 else None)
    full20 = r20["spectrum"] + (r20.get("spurious_exponents") or [])
    for k, v in enumerate(full20):
        ax.plot([2.6, 3.4], [v, v], color=C_BAD, lw=1.8 if k < 3 else 1.0,
                alpha=1.0 if k < 3 else .6, label="d=20" if k == 0 else None)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["truth", "d=3", "d=20 (top 8 shown)"])
    ax.set_ylabel("Lyapunov exponent"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Why d=20 fails: no spectral gap after the 3rd exponent\n"
                 "the extra latent directions form a continuum, so the 'top 3' is not the "
                 "real dynamics", fontsize=9.5)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_dim_ladder.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_dim_ladder")


if __name__ == "__main__":
    model, mu, sd = get_model(force="--retrain" in sys.argv)
    fig_attractor(model, mu, sd)
    fig_tracking(model, mu, sd)
    fig_convergence(model, mu, sd)
    fig_return_map(model, mu, sd)
    fig_off_ablation()
    fig_dim_ladder()
    print(f"\nfigures -> {M}")
