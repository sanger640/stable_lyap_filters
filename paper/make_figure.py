"""Figure 1 of the workshop paper: (a) one real fork as rendered endings, (b) real vs world-model futures
of one held-out fork state on the real futures' principal axes, (c) fragility agreement per panel.

Inputs are committed results or local caches: panel-3 example frames (scratch extraction is not used;
frames are read from results/program_b/visual_panel3), D0 latents and world models (results/program_b/step3),
and kappa values from results/jenga/*panel*_result.json.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
STATE = "ep63_seed104_s74"           # most balanced real split among held-out D0 forks (31/64 minority)
PANEL3_STATE = ("73", 34)            # panel-3 fork caught by every monitor; frames are aware-v1 endings
INK, MUTED, ORANGE, TEAL, VIOLET, GREY = "#1b2430", "#6b7682", "#d27a1e", "#0e7a86", "#6b55c9", "#a7b0b9"


def topple_flags(traces, start_tilt):
    from fresh_fragility_panel import topple_count
    return np.array([topple_count(traces[j:j + 1], start_tilt) > 0 for j in range(len(traces))])


def panel_a(axes):
    import fresh_fragility_panel as fp
    e, i = PANEL3_STATE
    d = ROOT / "results/program_b/visual_panel3"
    screen = json.loads((d / "screen.json").read_text())
    with np.load(d / f"ep{e}_s{i}.npz") as f, np.load(d / f"ep{e}_s{i}_frames.npz") as fr:
        flags = topple_flags(f["aware"], np.array(screen[e][str(i)]["start_tilt"]))
        up, down = int(np.flatnonzero(~flags)[0]), int(np.flatnonzero(flags)[0])
        for ax, j, label in ((axes[0], up, "stays"), (axes[1], down, "topples")):
            ax.imshow(fr["aware"][j, 1][38:98, 118:202], interpolation="lanczos")
            ax.set_title(label, fontsize=7, color=ORANGE if label == "topples" else TEAL, pad=2)
            ax.axis("off")
    return int(flags.sum())


def panel_b(ax):
    import torch
    import step3_data as sd
    import step3_diagnostics as dg
    import step3_world_model as wm
    paths = sorted((sd.OUT / "d0").glob("ep*.npz"))
    n = [p.stem for p in paths].index(STATE)
    with np.load(paths[n]) as f:
        real = f["z"][:, 1].reshape(64, -1).astype(np.float64)
        flags = topple_flags(f["traces"], f["start_tilt"])
        z0, acts = f["z0"][None].astype(np.float32), f["actions"]
    sto = wm.WorldModel.load(sd.OUT / "wm_stochastic.pt")
    det = wm.WorldModel.load(sd.OUT / "wm_deterministic.pt")
    s_end = dg.rollout(sto, z0, acts, "S_step", seed=20261060 + n)[1]
    d_end = dg.rollout(det, z0, acts, "D", seed=20261060 + n)[1]
    mean = real.mean(0)
    _, sv, vt = np.linalg.svd(real - mean, full_matrices=False)
    scale = sv[0] / np.sqrt(len(real))
    proj = lambda x: (x - mean) @ vt[:2].T / scale
    r, s, d = proj(real), proj(s_end), proj(d_end)
    ax.scatter(*s.T, s=7, c=VIOLET, alpha=0.55, lw=0, label="WM, stochastic")
    ax.scatter(*r[~flags].T, s=9, c=TEAL, lw=0, label="real, stays")
    ax.scatter(*r[flags].T, s=9, c=ORANGE, lw=0, label="real, topples")
    ax.scatter(*d.T, s=12, c=INK, marker="x", lw=0.8, label="WM, deterministic")
    ax.set_xlabel("real futures, PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left", ncol=2, handletextpad=0.1, columnspacing=0.4,
              borderaxespad=0.0, bbox_to_anchor=(-0.05, 1.0))
    torch.cuda.empty_cache()


def panel_c(ax):
    res = {2: json.loads((ROOT / "results/jenga/visual_pipeline_result.json").read_text()),
           3: json.loads((ROOT / "results/jenga/visual_pipeline_panel3_result.json").read_text()),
           4: json.loads((ROOT / "results/jenga/visual_v2_panel4_result.json").read_text())}
    keys = {2: {"naive": "naive", "aware v1": "aware", "aware v2": "aware_v2"},
            3: {"naive": "naive", "aware v1": "aware", "aware v2": "aware_v2", "v1 tokens": "aware_token"},
            4: {"naive": "naive", "aware v1": "aware", "aware v2": "v2_state", "v1 tokens": "v1_token",
                "v2 tokens": "v2_token"}}
    names = ["naive", "aware v1", "aware v2", "v1 tokens", "v2 tokens"]
    shown = {"naive": "no percep. sampling", "aware v1": "probe", "aware v2": "probe, 5 hyp.", "v1 tokens": "probe, images", "v2 tokens": "5 hyp., images"}
    colours = [GREY, TEAL, "#3fa7b3", VIOLET, "#9a8be0"]
    w = 0.16
    for p, panel in enumerate((2, 3, 4)):
        g = res[panel]["graded_vs_truth_A"]
        for m, name in enumerate(names):
            if name in keys[panel]:
                ax.bar(p + (m - 2) * w, g[keys[panel][name]]["kappa"], w * 0.92, color=colours[m],
                       label=shown[name] if p == 2 else None)
        ceil = res[panel]["ceiling_A_vs_B"]["kappa"]
        ax.hlines(ceil, p - 2.6 * w, p + 2.6 * w, colors=INK, lw=1)
    ax.set_xticks([0, 1, 2], ["set A", "set B", "set C"], fontsize=7)
    ax.set_ylabel(r"$\kappa$ vs truth", fontsize=7); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.plot([], [], color=INK, lw=1, label="ceiling (truth B vs A)")
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper left", columnspacing=0.6, handlelength=1,
              borderaxespad=0.0, bbox_to_anchor=(0.0, 1.03))


def main():
    plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(5.5, 1.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.62, 0.62, 1.1, 1.3], wspace=0.55)
    sub = gs[0, 0:2].subgridspec(1, 2, wspace=0.04)
    panel_a([fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])])
    axb, axc = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    panel_b(axb); panel_c(axc)
    for ax, letter, x, y in ((fig.axes[0], "(a)", -0.05, 1.25), (axb, "(b)", -0.32, 1.22), (axc, "(c)", -0.2, 1.22)):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.savefig(Path(__file__).with_name("fig_forks.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).with_name("fig_forks.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
