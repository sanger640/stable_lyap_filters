"""Figures for Phase 1 from results/phase1/*.json."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P1, M = ROOT / "results" / "phase1", ROOT / "media"
M.mkdir(exist_ok=True)
C_TRUE, C_OK, C_BAD, C_MID = "#333333", "#55A868", "#C44E52", "#4C72B0"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def fig_spectrum(tag="beta_sweep"):
    f = P1 / f"stage2_{tag}.json"
    if not f.exists():
        print(f"  skip: {f.name} not present"); return
    d = json.load(open(f)); truth = d["truth"]
    rows = [r for r in d["rows"] if r.get("ok")]
    if not rows: print("  skip: no successful rows"); return
    cfgs = sorted({r["config"] for r in rows}, key=lambda c: float(c.split("=")[-1]) if "=" in c else -1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    names = [r"$\lambda_1$ (chaos)", r"$\lambda_2$ (flow, =0)", r"$\lambda_3$ (contraction)"]
    for k, ax in enumerate(axes):
        mus = [np.mean([r["spectrum"][k] for r in rows if r["config"] == c]) for c in cfgs]
        sds = [np.std([r["spectrum"][k] for r in rows if r["config"] == c]) for c in cfgs]
        band = 0.05 * abs(truth[0] if k < 2 else truth[2])
        ax.axhspan(truth[k] - band, truth[k] + band, color=C_OK, alpha=.18,
                   label="±5% accept band")
        ax.axhline(truth[k], color=C_TRUE, ls="--", lw=1.4, label=f"truth {truth[k]:.3f}")
        cols = [C_OK if abs(m - truth[k]) <= band else C_BAD for m in mus]
        ax.bar(range(len(cfgs)), mus, yerr=sds, capsize=3, color=cols, edgecolor="white")
        for i, m in enumerate(mus):
            ax.text(i, m, f"{m:.2f}", ha="center",
                    va="bottom" if m > truth[k] else "top", fontsize=7.5)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([c.replace("beta=", "β=") for c in cfgs], fontsize=8)
        ax.set_title(names[k], fontsize=10)
        if k == 0: ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Phase 1 — learned Lorenz spectrum vs ground truth\n"
                 "λ₁ is easy; λ₃ is the exponent that reveals whether the Jacobians are right",
                 fontsize=10, y=1.04)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_spectrum.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_spectrum")


def fig_truth_check():
    f = P1 / "stage1_truth.json"
    if not f.exists(): print("  skip: stage1_truth.json"); return
    d = json.load(open(f))
    fig, ax = plt.subplots(figsize=(6.6, 2.6)); ax.axis("off")
    rows = [
        ["computed spectrum", str(d["spectrum"])],
        ["published reference", str(d["published"])],
        ["abs err vs published", str(d["abs_err_vs_published"])],
        ["sum of exponents", f"{d['sum']}"],
        ["must equal trace", f"{d['expected_sum_trace']}   (err {d['sum_error']})"],
        ["zero-exponent error", f"{d['closest_to_zero']}"],
        ["reference-free checks", "PASSED" if d["reference_free_checks_passed"] else "FAILED"],
    ]
    t = ax.table(cellText=rows, colWidths=[.42, .58], cellLoc="left", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.5)
    for i in range(len(rows)):
        t[(i, 0)].set_facecolor("#f2f2f2")
    t[(len(rows) - 1, 1)].set_facecolor("#dff0d8" if d["reference_free_checks_passed"] else "#f2dede")
    ax.set_title("Stage 1 — the measuring instrument, checked on the TRUE equations\n"
                 "sum-equals-trace and one-exponent-is-zero need no external reference",
                 fontsize=9.5, pad=14)
    fig.tight_layout(); fig.savefig(M / "fig_phase1_truth_check.png", bbox_inches="tight")
    plt.close(fig); print("  ok  fig_phase1_truth_check")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "beta_sweep"
    fig_truth_check(); fig_spectrum(tag)
