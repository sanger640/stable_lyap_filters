"""Figure 1 of the workshop paper.

Top: the fork probe as a pipeline, with real renders of one held-out fork (paper/fig_assets, made by
paper/render_fork_frames.py). Bottom: that fork's 64 real latent futures next to the two token world
models' futures for the same image and perturbed chunks, all on the real futures' principal axes.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
ASSETS = Path(__file__).with_name("fig_assets")
STATE = "ep63_seed104_s74"           # most balanced real split among held-out D0 forks (33/64 topple)
INK, MUTED, ORANGE, TEAL, VIOLET, BOX = "#1b2430", "#6b7682", "#d27a1e", "#0e7a86", "#6b55c9", "#eef1f4"


def crop(path):
    from PIL import Image
    return np.asarray(Image.open(path))[50:505, 55:465]


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


def box(ax, x, y, w, h, text, size=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2", fc=BOX, ec=MUTED, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED))


def image(fig, ax, x, y, w, h, img, label=None, colour=INK):
    """Place an image in the schematic axes' data coordinates."""
    (fx0, fy0), (fx1, fy1) = fig.transFigure.inverted().transform(ax.transData.transform([(x, y), (x + w, y + h)]))
    sub = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    sub.imshow(img, interpolation="lanczos", aspect="auto"); sub.set_xticks([]); sub.set_yticks([])
    for spine in sub.spines.values():
        spine.set_visible(True); spine.set_edgecolor(colour); spine.set_linewidth(0.8)
    if label:
        sub.set_title(label, fontsize=6, color=colour, pad=1.5)


def main():
    plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(5.5, 2.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.55, wspace=0.28)

    # ---- top: the probe as a pipeline (data coordinates 0..100 x 0..30; 1 x-unit ~ 1.37 y-units)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    image(fig, ax, 0, 4, 12, 16.4, crop(ASSETS / "start.png"), "camera image")
    arrow(ax, 12.6, 15.0, 12)
    box(ax, 15.8, 3.5, 17.5, 17, "estimate + measured\nperception error;\nchunk + measured\nexecution error\n= 64 imagined starts",
        size=5.6)
    arrow(ax, 34.3, 36.7, 12)
    box(ax, 37.5, 3.5, 15.5, 17, "rollout engine:\nMuJoCo (reference)\nor world model;\nchunk, then 30-step\nimagined hold", size=5.6)
    arrow(ax, 54.0, 56.4, 12)
    image(fig, ax, 57.2, 4, 12, 16.4, crop(ASSETS / "end_stays.png"), "stays: 31", TEAL)
    image(fig, ax, 70.2, 4, 12, 16.4, crop(ASSETS / "end_topples.png"), "topples: 33", ORANGE)
    ax.text(69.7, 0.6, "64 imagined endings", ha="center", va="center", fontsize=5.8, color=INK)
    arrow(ax, 83.0, 85.4, 12)
    box(ax, 86.2, 3.5, 13.0, 17, "split into two\npersistent,\nseparated\ngroups?\nyes = fragile", size=5.6)
    ax.text(0, 30.5, "(a) the fork probe", fontsize=7.5, fontweight="bold", va="top")

    # ---- bottom: real vs world-model futures for the same state
    real, flags, sto, det = futures()
    panels = (("real futures (physics)", None), ("world model, stochastic", sto),
              ("world model, deterministic", det))
    for c, (title, pts) in enumerate(panels):
        a = fig.add_subplot(gs[1, c])
        if pts is None:
            a.scatter(*real[~flags].T, s=6, c=TEAL, lw=0, label="stays")
            a.scatter(*real[flags].T, s=6, c=ORANGE, lw=0, label="topples")
            a.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, handletextpad=0.1,
                     columnspacing=0.6, borderaxespad=0.1)
        else:
            a.scatter(*real.T, s=6, c="#c9cfd5", lw=0, label="real")
            a.scatter(*pts.T, s=6, c=VIOLET if c == 1 else INK, lw=0, label="model")
            a.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, handletextpad=0.1,
                     columnspacing=0.6, borderaxespad=0.1)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-0.9, 1.35)
        a.set_title(title, fontsize=6.5, pad=2)
        a.tick_params(labelsize=5.5, length=2)
        if c == 0:
            a.set_ylabel("PC2", fontsize=6)
            a.text(-0.42, 1.2, "(b)", transform=a.transAxes, fontsize=7.5, fontweight="bold")
        a.set_xlabel("PC1 of real futures", fontsize=6)
    fig.savefig(Path(__file__).with_name("fig_forks.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).with_name("fig_forks.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
