"""REDM1 (results/jenga/redm1_protocol.json, offline, CPU): is the red block hidden from the irreversibility score by the
hand mask?
For CF3's 4679 fresh decisions (policy alone, 2x): (M1) the share of the red block's image footprint that lies inside the
hand mask CF3 used, and (M2) whether CF3's stored score r_max separates decisions 1-3 chunks before a red-block fall from
clean safe decisions.
usage: python3 eval/redm1.py -> results/jenga/redm1_result.json
       python3 eval/redm1.py nm -> results/jenga/nm1_stage0.json (NM1 stage 0: the same grading on the no-mask CF3 scores,
       results/jenga/cf3nm_rows_*.npz, next to the masked ones; report only)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUNS = ("paired_record_s4", "paired_record_h2h2")                 # = eval/cf_conf.RUNS (index order of cf3_rows 'run')
HJ, PR = ROOT / "results/jenga/hj2/data", ROOT / "results/program_b/policy_rollouts"
TAU, S0, NL, FALL, EXT = 1.3356, 1, 15, 45.0, 0.02               # CF3 threshold, plan slice (cbf_filter.CFG), tilt, 2 cm


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def box_share(sim, body, mask, cb):
    """share of the patches covered by the body's projected box (patch centre inside the hull of the 8 corners) that are
    inside the mask; also the number of covered patches"""
    from matplotlib.path import Path as MPath
    from scipy.spatial import ConvexHull
    g = int(np.where(sim.model.geom_bodyid == body)[0][0])
    h, R, c = sim.model.geom_size[g], sim.data.geom_xmat[g].reshape(3, 3), sim.data.geom_xpos[g]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    uv = cb.project(c + corners @ R.T)
    hull = MPath(uv[ConvexHull(uv).vertices])
    ctr = (np.arange(cb.GRID) + 0.5) * cb.IMG / cb.GRID
    U, V = np.meshgrid(ctr, ctr)
    cov = hull.contains_points(np.c_[U.ravel(), V.ravel()]).reshape(cb.GRID, cb.GRID)
    return (float((cov & mask).sum() / cov.sum()) if cov.sum() else float("nan")), int(cov.sum())


def main():
    import cbf_barrier as cb
    import inspector_data as insp
    rows = {k: np.concatenate([np.load(ROOT / f"results/jenga/cf3_rows_{s}.npz")[k] for s in (0, 1)])
            for k in ("run", "ep", "i", "kt", "k", "closed", "r")}
    sim = insp._sim()
    ids = [sim.model.body(n).id for n in ("block_middle", "block_left", "block_right")]
    eps = sorted({(int(r), int(e)) for r, e in zip(rows["run"], rows["ep"])})
    red_k, cache = {}, {}
    for r, e in eps:                                              # red-fall chunk per episode (saved chunk-start states)
        S = np.load(PR / RUNS[r] / f"ep{e}.npz")
        H = np.load(HJ / RUNS[r] / f"ep{e}.npz")
        tilt, ext = [], []
        p0 = None
        for d in range(len(S["chunk_index"])):
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
            p = sim.data.xpos[ids[0]].copy(); p0 = p if p0 is None else p0
            tilt.append(float(sim.block_diagnostics()["tilt"][0]))
            ext.append(bool(p[2] - p0[2] >= EXT or p[0] - p0[0] <= -EXT))
        kr = -1
        if int(H["topple_chunk"]) < 0:
            for d in range(len(tilt) - 1):
                if any(ext[:d + 1]):
                    break
                if tilt[d] < FALL <= tilt[d + 1]:
                    kr = int(S["chunk_index"][d]); break
        red_k[(r, e)] = kr
        cache[(r, e)] = (S, H)
    # M1: geometry at every CF3 decision
    share = np.full((len(rows["ep"]), 3), np.nan); npatch = np.zeros((len(rows["ep"]), 3), int)
    for n, (r, e, i) in enumerate(zip(rows["run"], rows["ep"], rows["i"])):
        S, H = cache[(int(r), int(e))]
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        pts = np.vstack([H["proprio"][i, :3][None], np.asarray(S["plan16"][i], np.float32)[S0:S0 + NL, :3]])
        m = cb.hand_mask(pts)
        for b in range(3):
            share[n, b], npatch[n, b] = box_share(sim, ids[b], m, cb)
    kr = np.array([red_k[(int(r), int(e))] for r, e in zip(rows["run"], rows["ep"])])
    pre_top = (rows["kt"] >= 0) & (rows["kt"] - rows["k"] >= 1) & (rows["kt"] - rows["k"] <= 3)
    pre_red = (kr >= 0) & (kr - rows["k"] >= 1) & (kr - rows["k"] <= 3)
    clean = (rows["kt"] < 0) & (kr < 0)
    rmax = rows["r"].max(1)
    closed = rows["closed"] > 0

    def m1(sel):
        return {"n": int(sel.sum()), "red_share_masked_median": float(np.nanmedian(share[sel, 0])),
                "red_share_masked_quartiles": [float(np.nanpercentile(share[sel, 0], q)) for q in (25, 75)],
                "red_fully_masked_frac": float(np.nanmean(share[sel, 0] >= 0.999)),
                "neighbour_share_masked_median": float(np.nanmedian(share[sel, 1:])),
                "red_patches_median": float(np.median(npatch[sel, 0]))}

    def m2(sel):
        return {"n": int(sel.sum()), "auc_vs_clean_safe": auc(rmax[sel], rmax[clean]),
                "recall_at_tau": float((rmax[sel] > TAU).mean()), "recall_by_lead": [
                    float((rmax[sel & (lead == L)] > TAU).mean()) if (sel & (lead == L)).any() else None for L in (1, 2, 3)]}
    lead = np.where(pre_red, kr - rows["k"], np.where(pre_top, rows["kt"] - rows["k"], 0))
    res = {"episodes": len(eps), "red_fall_episodes_before_extraction_no_topple": int(sum(v >= 0 for v in red_k.values())),
           "M1": {"pre_red": m1(pre_red), "pre_topple": m1(pre_top), "clean_safe_closed": m1(clean & closed),
                  "clean_safe_open": m1(clean & ~closed)},
           "M2": {"pre_red": m2(pre_red), "pre_topple": m2(pre_top),
                  "clean_safe_flag_rate": {"closed": float((rmax[clean & closed] > TAU).mean()),
                                           "open": float((rmax[clean & ~closed] > TAU).mean())}}}
    n, med, a = res["M2"]["pre_red"]["n"], res["M1"]["pre_red"]["red_share_masked_median"], res["M2"]["pre_red"]["auc_vs_clean_safe"]
    res["verdict"] = ("DESCRIPTIVE_ONLY (n < 30)" if n < 30 else "SCORE_SEES" if a >= 0.65 else
                      "MASK_BLINDS" if med >= 0.5 else "NOT_MASK")
    np.savez_compressed(ROOT / "results/jenga/redm1_rows.npz", kr=kr, share=share, npatch=npatch, rmax=rmax, kt=rows["kt"],
                        k=rows["k"], closed=rows["closed"], run=rows["run"], ep=rows["ep"])
    (ROOT / "results/jenga/redm1_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage0():
    Q = np.load(ROOT / "results/jenga/redm1_rows.npz")
    N = {k: np.concatenate([np.load(ROOT / f"results/jenga/cf3nm_rows_{s}.npz")[k] for s in (0, 1)]) for k in ("run", "ep", "k", "r")}
    key = {(int(a), int(b), int(c)): n for n, (a, b, c) in enumerate(zip(N["run"], N["ep"], N["k"]))}
    idx = np.array([key[(int(a), int(b), int(c))] for a, b, c in zip(Q["run"], Q["ep"], Q["k"])])
    kr, kt, k, cl = Q["kr"], Q["kt"], Q["k"], Q["closed"] > 0
    pre_red = (kr >= 0) & (kr - k >= 1) & (kr - k <= 3); pre_top = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); clean = (kt < 0) & (kr < 0)
    out = {}
    for name, r in (("masked", Q["rmax"]), ("no_mask", N["r"][idx].max(1))):
        out[name] = {"pre_red": {"auc_vs_clean_safe": auc(r[pre_red], r[clean]), "recall_at_tau": float((r[pre_red] > TAU).mean()),
                                 "median": float(np.median(r[pre_red]))},
                     "pre_topple": {"auc_vs_clean_safe": auc(r[pre_top], r[clean]), "recall_at_tau": float((r[pre_top] > TAU).mean())},
                     "clean_safe_flag_rate": {"closed": float((r[clean & cl] > TAU).mean()), "open": float((r[clean & ~cl] > TAU).mean())},
                     "clean_closed_median": float(np.median(r[clean & cl]))}
    (ROOT / "results/jenga/nm1_stage0.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    stage0() if sys.argv[1:] == ["nm"] else main()
