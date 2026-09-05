"""
Generate every figure in the README from results/*.json. Deterministic, no GPU, no model.

    python eval/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R, M = ROOT / "results", ROOT / "media"
M.mkdir(exist_ok=True)

# colourblind-safe, consistent across every figure
C_SAFE, C_UNSAFE = "#4C72B0", "#C44E52"
C_GOOD, C_MID, C_BAD, C_ACCENT = "#55A868", "#DD8452", "#8C8C8C", "#8172B3"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def load(n):
    return json.load(open(R / f"{n}.json"))


def auc(pos, neg):
    p, n = np.asarray(pos, float), np.sort(np.asarray(neg, float))
    r = (np.searchsorted(n, p, "left") +
         0.5 * (np.searchsorted(n, p, "right") - np.searchsorted(n, p, "left")))
    return float(r.mean() / len(n))


def roc(pos, neg):
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = [(np.asarray(pos) >= t).mean() for t in thr]
    fpr = [(np.asarray(neg) >= t).mean() for t in thr]
    return np.array(fpr), np.array(tpr)


# ---------------------------------------------------------------- 1. masking ladder
def fig_masking():
    d = load("pc1_mask_full_corpus")
    stages = ["unmasked", "low_norm", "pc1"]
    labels = ["row mask\nonly", "+ low-norm\n(k=30)", "+ PC1 background\n(75% keep)"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(stages))
    for i, (metric, runs) in enumerate(d["results"].items()):
        mu = [np.mean([r[s] for r in runs]) for s in stages]
        sd = [np.std([r[s] for r in runs]) for s in stages]
        ax.bar(x + (i - 0.5) * 0.36, mu, 0.34, yerr=sd, capsize=3,
               label=metric, color=[C_MID, C_ACCENT][i], edgecolor="white")
        for xi, m, e in zip(x + (i - 0.5) * 0.36, mu, sd):
            ax.text(xi, m + e + 0.012, f"{m:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, ls=":", c=C_BAD, lw=1)
    ax.text(2.42, 0.515, "chance", fontsize=7, c=C_BAD, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("held-out AUC"); ax.set_ylim(0.45, 1.0); ax.legend(frameon=False)
    ax.set_title("Patch masking drives the result\n1772 chunks · 20 held-out episode splits",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(M / "fig_masking_ladder.png"); plt.close(fig)


# ------------------------------------------------- 2. score distributions + ROC
def fig_separation():
    d = load("nominal_baseline")
    s = np.array([c["nominal"]["p90"] for c in d["chunks"]])
    y = np.array([c["y"] for c in d["chunks"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    bins = np.linspace(0, np.percentile(s, 99.5), 45)
    a1.hist(s[y == 0], bins=bins, color=C_SAFE, alpha=.85, label=f"safe (n={(y==0).sum()})",
            density=True)
    a1.hist(s[y == 1], bins=bins, color=C_UNSAFE, alpha=.8, label=f"unsafe (n={(y==1).sum()})",
            density=True)
    thr = np.percentile(s[y == 0], 95)
    a1.axvline(thr, c="k", ls="--", lw=1.2)
    a1.text(thr, a1.get_ylim()[1] * .82, "  p95 of safe\n  (threshold)", fontsize=7.5)
    a1.set_xlabel("monitor score  (d_end, p90 over patches)")
    a1.set_ylabel("density"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("Score separation", fontsize=10)

    fpr, tpr = roc(s[y == 1], s[y == 0])
    a2.plot(fpr, tpr, c=C_ACCENT, lw=2, label=f"AUC = {auc(s[y==1], s[y==0]):.3f}")
    a2.plot([0, 1], [0, 1], ls=":", c=C_BAD, lw=1, label="chance")
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.legend(frameon=False, fontsize=8); a2.set_title("ROC", fontsize=10)
    fig.suptitle("The threshold is set from SAFE chunks only — no failure labels used",
                 fontsize=9.5, y=1.02)
    fig.tight_layout(); fig.savefig(M / "fig_separation.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------- 3. the phase confound
def fig_confound():
    d = load("phase_confound_test")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    groups = [k for k in d if k.startswith("d_end")]
    lbl = ["foreground\n(arm, gripper, moving block)", "background\n(table, static blocks)"]
    x = np.arange(2); w = 0.27
    for j, (key, colour) in enumerate([("corr_all", C_MID), ("corr_safe", C_SAFE),
                                       ("corr_unsafe", C_UNSAFE)]):
        vals = [d[g][key] if np.isfinite(d[g][key]) else 0 for g in groups]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=key.replace("corr_", ""),
                      color=colour, edgecolor="white")
        for b, g in zip(bars, groups):
            v = d[g][key]
            ax.text(b.get_x() + b.get_width() / 2,
                    (v if np.isfinite(v) else 0) + .012,
                    "n/a" if not np.isfinite(v) else f"{v:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(lbl); ax.axhline(0, c="k", lw=.8)
    ax.set_ylabel("corr(ground-truth patch motion, d_end)")
    ax.legend(frameon=False, fontsize=8, title="chunks", title_fontsize=8)
    ax.set_title("Why the BACKGROUND carries the signal\n"
                 "foreground divergence tracks motion just as strongly in SAFE chunks\n"
                 "→ it is a motion confound, not a failure signal", fontsize=9.5)
    fig.tight_layout(); fig.savefig(M / "fig_phase_confound.png"); plt.close(fig)


# ------------------------------------------- 4. operating points vs the probe
def fig_operating():
    d = load("probe_vs_divergence")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    cols = {"probe (tilt at t+8)": C_ACCENT, "d_end p90 / k=30": C_MID,
            "nominal p90 / k=30": C_BAD}
    for name, v in d["results"].items():
        pts = v.get("points", {})
        if not pts:
            continue
        order = sorted(pts, key=lambda k: -pts[k]["recall"])
        rec = [pts[k]["recall"] for k in order]
        pre = [pts[k]["precision"] for k in order]
        c = cols.get(name, "k")
        a1.plot(rec, pre, "o-", color=c, lw=1.6, ms=4,
                label=f"{name}  (AUC {v['auc']:.3f})")
        for k in order:
            a1.annotate(k, (pts[k]["recall"], pts[k]["precision"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", color=c)
        a2.plot([pts[k]["recall"] for k in order], [pts[k]["f1"] for k in order],
                "o-", color=c, lw=1.6, ms=4)
    a1.set_xlabel("recall"); a1.set_ylabel("precision")
    a1.legend(frameon=False, fontsize=7.5); a1.set_title("Operating curve", fontsize=10)
    a2.set_xlabel("recall"); a2.set_ylabel("F1"); a2.set_title("F1 vs recall", fontsize=10)
    fig.suptitle("A supervised tilt probe beats zero-shot divergence — "
                 "but needs labels the monitor never uses", fontsize=9.5, y=1.02)
    fig.tight_layout(); fig.savefig(M / "fig_operating_points.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------- 5. PC1 sign: which side wins
def fig_sign():
    d = load("pc1_sign_sweep_compare")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    keys = [k for k in d if "ensemble" not in k]
    names = [k.replace(" (sign=+1)", "").replace(" (sign=-1)", "") for k in keys]
    vals = [d[k]["mean"] for k in keys]
    errs = [d[k]["std"] for k in keys]
    cols = [C_GOOD if "background" in k else C_MID for k in keys]
    yy = np.arange(len(keys))
    ax.barh(yy, vals, xerr=errs, color=cols, edgecolor="white", capsize=3)
    for i, (v, k) in enumerate(zip(vals, keys)):
        pct = max(d[k]["pct_chosen"].items(), key=lambda x: x[1])
        ax.text(v + .012, i, f"{v:.3f}   (picked {pct[0]}% in {pct[1]}/20)",
                va="center", fontsize=7.5)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.5, 1.06); ax.set_xlabel("held-out AUC")
    ax.axvline(0.5, ls=":", c=C_BAD, lw=1)
    ax.set_title("Keeping the LOW-motion (background) patches wins\n"
                 "the foreground direction's own optimum is 'no filtering at all'",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(M / "fig_pc1_sign.png"); plt.close(fig)


# ------------------------------------------------ 6. perturbation magnitude sweep
def fig_sigma():
    d = load("sigma_sweep")
    keys = sorted(d, key=float)
    sig = [float(k) for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for field, c, lbl in [("dend", C_ACCENT, "d_end (50 perturbations)"),
                          ("nominal", C_MID, "nominal (1 rollout)")]:
        v = []
        for k in keys:
            y = np.asarray(d[k]["y"]); s = np.asarray(d[k][field], float)
            v.append(auc(s[y == 1], s[y == 0]))
        ax.plot(sig, v, "o-", c=c, lw=1.8, ms=5, label=lbl)
    ax.axvline(0.05, ls="--", c=C_BAD, lw=1)
    ax.text(0.052, ax.get_ylim()[0] + .02, "operating σ", fontsize=7, c=C_BAD)
    ax.set_xlabel("perturbation σ"); ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Perturbation magnitude sweep", fontsize=10)
    fig.tight_layout(); fig.savefig(M / "fig_sigma_sweep.png"); plt.close(fig)


# ---------------------------------------------------- 7. what the patch masks keep
def fig_patch_masks():
    """Overlay the geometric row mask on a real encoder-input frame.

    Uses the bundled media/_frame_224.png (already resized+cropped exactly as the encoder
    sees it), so this needs no dataset access."""
    import cv2
    src = M / "_frame_224.png"
    if not src.exists():
        raise FileNotFoundError(src)
    img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    G, P = 14, 16                      # 14x14 patches of 16px on the 224px input
    masked_rows = (0, 1, 8, 9, 10, 11, 12, 13)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, show_mask in zip(axes, (False, True)):
        ax.imshow(img)
        for r in range(G):
            for c in range(G):
                if show_mask and r in masked_rows:
                    ax.add_patch(plt.Rectangle((c * P, r * P), P, P, facecolor=C_UNSAFE,
                                               alpha=.42, edgecolor="none"))
                ax.add_patch(plt.Rectangle((c * P, r * P), P, P, fill=False,
                                           edgecolor="white", lw=.3, alpha=.5))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_title("encoder input\n14×14 = 196 DINOv2 patches", fontsize=9.5)
    axes[1].set_title("geometric row mask\nred = discarded (rows 0–1, 8–13)", fontsize=9.5)
    fig.suptitle("Step 1 of masking: drop ceiling and the high-texture checkered floor.\n"
                 "84 of 196 patches survive; PC1 then keeps the low-motion 75% of those.",
                 fontsize=9, y=1.0)
    fig.tight_layout(); fig.savefig(M / "fig_patch_masks.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_masking, fig_separation, fig_confound, fig_operating, fig_sign,
               fig_sigma, fig_patch_masks):
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nfigures -> {M}")
