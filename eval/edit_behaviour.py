"""Descriptive (not a registered test): what each filter does to the policy's moves in H2H5.

For every 0.1 s target of every saved decision (results/program_b/policy_rollouts/<run>/ep*.npz) the edit is
executed - nominal (mm). It is expressed in the task frame read off the saved simulator state at that decision:
  x  = along the pull axis (+ = toward the blocks), lat = sideways, + = toward the neighbour the hand is nearer to,
  z  = up.
Phase: reach (gripper open, hand > 40 mm behind the red block's centre), near (open, <= 40 mm), closed (close command).
Topple timing comes from each episode's json log (first chunk with topple_started).

usage: python3 eval/edit_behaviour.py [2x|1x]   -> results/jenga/edit_behaviour_<nz>.npz (cache) + figures
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUNS = ROOT / "results/program_b/policy_rollouts"
NZ = sys.argv[1] if len(sys.argv) > 1 else "2x"
ARMS = {"policy": f"paired_record_h5_{NZ}", "ours90": f"paired_lfc000_h5o90_{NZ}",
        "ours80": f"paired_lfc000_h5o80_{NZ}", "unisafe": f"paired_lsfa030_h5_{NZ}"}
EPS = range(26200, 26800)
EDIT_MIN = 0.5                                   # mm, as in eval/filter_videos.py
NEAR = 40.0                                      # mm behind the red block's centre
CACHE = ROOT / f"results/jenga/edit_behaviour_{NZ}.npz"


def extract():
    import inspector_data as insp
    sim = insp._sim()
    rows = []                                    # arm, ep, chunk, j, chunks_to_topple, phase, dx, dlat, dz, |d|, lat_nom, gap_nom
    for ai, (arm, run) in enumerate(ARMS.items()):
        for ep in EPS:
            S = np.load(RUNS / run / f"ep{ep}.npz")
            log = json.loads((RUNS / run / f"ep{ep}.json").read_text())["log"]
            tk = next((c["chunk"] for c in log if c.get("topple_started")), -1)
            for d in range(len(S["chunk_index"])):
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                P = sim.block_diagnostics()["position"] * 1000          # red, left, right (mm)
                k = int(S["chunk_index"][d])
                nom, exe = S["nominal"][d].astype(np.float64), S["executed"][d].astype(np.float64)
                for j in range(len(nom)):
                    p = nom[j, :3] * 1000
                    side = 1.0 if p[1] >= P[0, 1] else -1.0
                    nb = P[2] if side > 0 else P[1]
                    dv = (exe[j, :3] - nom[j, :3]) * 1000
                    phase = 2 if nom[j, 3] > 0 else (1 if P[0, 0] - p[0] <= NEAR else 0)
                    rows.append((ai, ep, k, j, (tk - k) if tk >= 0 else 999, phase, dv[0], dv[1] * side, dv[2],
                                 np.linalg.norm(dv), (p[1] - P[0, 1]) * side, abs(nb[1] - p[1])))
        print(arm, len(rows), flush=True)
    R = np.array(rows, np.float64)
    np.savez_compressed(CACHE, R=R, arms=np.array(list(ARMS)))
    return R


COL = {"ours90": "#3E9B4F", "ours80": "#2AA198", "unisafe": "#3A6FC4"}
LAB = {"ours90": "Ours, 90% barrier", "ours80": "Ours, 80% barrier", "unisafe": "UNISafe"}
PH = ["reaching", "at the block", "gripper closed"]


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    arms = list(ARMS)
    F = ("ours90", "ours80", "unisafe")
    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)
    S = {}
    # (a) how often, by phase
    ax = fig.add_subplot(gs[0, 0:2])
    w = 0.26
    for i, a in enumerate(F):
        A = R[R[:, 0] == arms.index(a)]
        r = [100 * (A[A[:, 5] == p, 9] > EDIT_MIN).mean() for p in range(3)]
        S[a] = {"edited_pct_by_phase": dict(zip(PH, r))}
        ax.bar(np.arange(3) + (i - 1) * w, r, w, color=COL[a], label=LAB[a])
    ax.set_xticks(range(3), PH); ax.set_ylabel("% of 0.1 s steps edited"); ax.set_title("(a) How often it edits", loc="left")
    ax.legend(frameon=False, fontsize=10)
    # (b) how big
    ax = fig.add_subplot(gs[0, 2:4])
    bins = np.logspace(np.log10(EDIT_MIN), np.log10(300), 40)
    for a in F:
        A = R[(R[:, 0] == arms.index(a)) & (R[:, 9] > EDIT_MIN)]
        S[a]["edit_mm_median"], S[a]["edit_mm_p90"] = float(np.median(A[:, 9])), float(np.percentile(A[:, 9], 90))
        ax.hist(A[:, 9], bins, histtype="step", lw=2.2, color=COL[a], density=True,
                label=f"{LAB[a]}: median {S[a]['edit_mm_median']:.0f} mm, 90th pct {S[a]['edit_mm_p90']:.0f} mm")
    ax.axvline(10, color="#888", ls=":", lw=1.2); ax.text(10.5, ax.get_ylim()[1] * 0.12, "1 cm", color="#666", fontsize=10)
    ax.set_xscale("log"); ax.set_xlabel("size of the change to a 0.1 s target (mm)"); ax.set_yticks([])
    ax.set_title("(b) How big the edits are", loc="left"); ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    # (c) sideways component at the block
    lim = 15
    ax = fig.add_subplot(gs[0, 4:6])
    for a in F:
        A = R[(R[:, 0] == arms.index(a)) & (R[:, 9] > EDIT_MIN) & (R[:, 5] == 1)]
        ax.hist(np.clip(A[:, 7], -lim, lim), np.linspace(-lim, lim, 41), histtype="step", lw=2.2, color=COL[a], density=True, label=LAB[a])
    ax.set_yticks([]); ax.set_xlabel("sideways change (mm)\n← toward the red block's centre | toward the nearer neighbour →")
    ax.set_title("(c) Sideways part of edits at the block\n(end bins collect everything beyond 15 mm)", loc="left")
    # (d) when, aligned to the policy's own topple in the same episode; (e) aligned to the arm's own topples
    P = R[R[:, 0] == 0]
    first = {int(e): int(k + t) for e, k, t in P[:, [1, 2, 4]] if t < 999}
    rel_r = np.arange(-10, 1)
    axd, axe = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6], sharey=None)
    for a in F:
        A = R[R[:, 0] == arms.index(a)]
        rel = np.array([k - first.get(int(e), 10 ** 6) for e, k in A[:, [1, 2]]])
        y = [100 * (A[rel == r, 9] > EDIT_MIN).mean() for r in rel_r]
        base = 100 * (A[A[:, 4] == 999, 9] > EDIT_MIN).mean()
        y2 = [100 * (A[A[:, 4] == -r, 9] > EDIT_MIN).mean() for r in rel_r]
        own = len({int(e) for e, t in A[:, [1, 4]] if t < 999})
        S[a]["edited_pct_before_policy_topple"] = dict(zip(map(int, rel_r), y))
        S[a]["edited_pct_before_own_topple"] = dict(zip(map(int, rel_r), y2))
        S[a]["edited_pct_never_toppled"] = base
        for ax, yy, lab in ((axd, y, LAB[a]), (axe, y2, f"{LAB[a]} ({own} topples)")):
            ax.plot(rel_r * 0.8, yy, "-o", color=COL[a], lw=2.4, ms=5, label=lab)
            ax.axhline(base, color=COL[a], ls=":", lw=1.2)
    axd.set_xlabel("seconds before the policy alone toppled a block (same start, same noise)"); axd.set_ylabel("% of steps edited")
    axd.set_title(f"(d) The {len(first)} episodes where the policy alone topples", loc="left")
    axe.set_xlabel("seconds before this filter's own topple"); axe.set_ylabel("% of steps edited")
    axe.set_title("(e) The episodes where the filter itself still topples", loc="left")
    for ax in (axd, axe):
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.text(-8, 0.5 + max(S[a]["edited_pct_never_toppled"] for a in F), "dotted: episodes that never topple", fontsize=9.5, color="#555")
    # (f-h) which way, at the block (side view: pull axis vs up)
    for i, a in enumerate(F):
        A = R[(R[:, 0] == arms.index(a)) & (R[:, 9] > EDIT_MIN) & (R[:, 5] == 1)]
        out = 100 * (np.abs(A[:, [6, 8]]) > lim).any(1).mean()
        S[a]["near_block_mean_dx_dlat_dz"] = [float(v) for v in A[:, [6, 7, 8]].mean(0)]
        S[a]["near_block_frac_back_up"] = [float((A[:, 6] < 0).mean()), float((A[:, 8] > 0).mean())]
        S[a]["near_block_pct_off_scale"] = out
        ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2])
        ax.hist2d(A[:, 6], A[:, 8], bins=60, range=[[-lim, lim], [-lim, lim]], cmap="Greens" if a != "unisafe" else "Blues",
                  norm=LogNorm(), cmin=1)
        ax.axhline(0, color="#aaa", lw=0.8); ax.axvline(0, color="#aaa", lw=0.8); ax.set_aspect("equal")
        ax.set_xlabel("change along the pull axis (mm)\n← back away | toward the blocks →")
        ax.set_ylabel("vertical change (mm)\n← down | up →")
        ax.set_title(f"({'fgh'[i]}) {LAB[a]}, edits at the block\n{100*(A[:,6]<0).mean():.0f}% back away, {100*(A[:,8]>0).mean():.0f}% go up"
                     + (f"; {out:.0f}% off the plot" if out >= 1 else ""), loc="left")
    fig.suptitle(f"What the filters do to the policy's moves (600 paired episodes, {NZ} noise; an edit = target moved > {EDIT_MIN} mm)",
                 fontsize=14, x=0.01, ha="left")
    out = ROOT / f"results/jenga/edit_behaviour_{NZ}.png"
    fig.savefig(out, dpi=130); print(out)
    (ROOT / f"results/jenga/edit_behaviour_{NZ}.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    R = np.load(CACHE)["R"] if CACHE.exists() else extract()
    figure(R)
