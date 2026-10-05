"""Figure 1 (b): one held-out fork's 64 real latent futures next to the two token world models' futures
for the same image and perturbed chunks, on the real futures' principal axes. Panel (a) is TikZ
(paper/fig_probe.tex) with renders from paper/render_fork_frames.py.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
STATE = "ep63_seed104_s74"           # most balanced real split among held-out D0 forks (33/64 topple)
INK, MUTED, LINE, ORANGE, TEAL, VIOLET, GHOST = "#1b2430", "#5d6873", "#c3cad1", "#c96a14", "#0e7a86", "#6b55c9", "#d5dade"


def futures():
    import step3_data as sd
    import step3_diagnostics as dg
    import step3_world_model as wm
    from fresh_fragility_panel import topple_count
    paths = sorted((sd.OUT / "d0").glob("ep*.npz"))
    n = [p.stem for p in paths].index(STATE)
    with np.load(paths[n]) as f:
        real = f["z"][:, 1].reshape(64, -1).astype(np.float64)
        flags = np.array([topple_count(f["traces"][j:j + 1], f["start_tilt"]) > 0 for j in range(64)])
        z0, acts = f["z0"][None].astype(np.float32), f["actions"]
    sto = dg.rollout(wm.WorldModel.load(sd.OUT / "wm_stochastic.pt"), z0, acts, "S_step", seed=20261060 + n)[1]
    det = dg.rollout(wm.WorldModel.load(sd.OUT / "wm_deterministic.pt"), z0, acts, "D", seed=20261060 + n)[1]
    mean = real.mean(0)
    _, sv, vt = np.linalg.svd(real - mean, full_matrices=False)
    scale = sv[0] / np.sqrt(len(real))
    proj = lambda x: (x - mean) @ vt[:2].T / scale
    return proj(real), flags, proj(sto), proj(det)


def main():
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.edgecolor": LINE,
                         "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    real, flags, sto, det = futures()
    fig, axes = plt.subplots(1, 3, figsize=(5.45, 1.25), sharey=True)
    specs = [("real futures (exact physics)", None, None, "two groups"),
             ("world model, stochastic", sto, VIOLET, "one cluster"),
             ("world model, deterministic", det, INK, "no gap")]
    for a, (title, pts, colour, verdict) in zip(axes, specs):
        if pts is None:
            a.scatter(*real[~flags].T, s=7, c=TEAL, lw=0)
            a.scatter(*real[flags].T, s=7, c=ORANGE, lw=0)
        else:
            a.scatter(*real.T, s=7, c=GHOST, lw=0)
            a.scatter(*pts.T, s=7, c=colour, lw=0)
        a.set_xlim(-1.55, 1.55); a.set_ylim(-0.95, 1.15)
        a.set_xticks([-1, 0, 1]); a.set_yticks([-0.5, 0, 0.5, 1])
        a.tick_params(labelsize=6.5, length=2, width=0.5)
        a.set_title(title, fontsize=8, color=INK, pad=3)
        a.text(0.5, 0.05, verdict, transform=a.transAxes, ha="center", va="bottom", fontsize=7.5,
               color=colour or INK, style="italic")
    axes[0].set_ylabel("PC 2", fontsize=7, color=MUTED, labelpad=1)
    axes[1].set_xlabel("PC 1 of the real futures (same axes in all panels)", fontsize=7, color=MUTED, labelpad=1)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=c, label=l) for c, l in
               ((TEAL, "real: neighbour stays"), (ORANGE, "real: neighbour topples"), (GHOST, "real (behind model)"),
                (VIOLET, "stochastic model"), (INK, "deterministic model"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, -0.30), handletextpad=0.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.12)
    fig.text(0.035, 1.0, "(b)", fontsize=9, fontweight="bold", color=INK, ha="right", va="bottom")
    fig.savefig(Path(__file__).with_name("fig_futures.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).with_name("fig_futures.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
