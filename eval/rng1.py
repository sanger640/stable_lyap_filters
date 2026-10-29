"""RNG1 stage R (results/jenga/rng1_protocol.json, offline): give the undo-test score a range, then calibrate its threshold
directly on undo-vs-hold scores.
Why: the current score r = max_j q98_outside-hand d_p (d_p = mean over the 8 latent channels of |f_rev_j - f_hold_j| per
patch) is bounded by how different two patch contents can be (channels have unit scale; two unrelated scenes differ by
~1.26 at the 98th patch percentile), so it saturates near 1.46 and tau = 1.3356 sits on a cliff (TAU1). Candidates that grow
with the AREA of change instead:
  S1 area: max_j share of outside-hand patches with d_p > c
  S2 mass: max_j mean over outside-hand patches of max(0, d_p - c)
  c   = pooled 99.5th percentile of d_p over the calibration undo tests (all patches, draws, decisions), rounded UP to GRID
  tau = conformal 98% of the score over the calibration decisions (flag at most 2% of the policy's own moves)
Calibration (label-free): 3000 seeded decisions of paired_record_ocl (policy alone, dev set, never a CF3 / H2H decision),
the policy's real first 8 control steps (0.8 s), undone, vs hold; 64 draws at 2x noise, exactly CF3's undo test.
Evaluation: CF3's 4679 fresh decisions (eval/cf_conf.py select(), labels for grading only), the same undo test (= the
stored irrm1 r8 rows, reproduced).
usage: SHARD=i/n python3 eval/rng1.py calib|cf3 -> results/jenga/rng1_{calib,cf3}_<i>.npz;  python3 eval/rng1.py grade
       -> results/jenga/rng1_result.json"""
import json
import os
from math import ceil
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga"
S, NL, TAU0, ALPHA, Q_C, N_CAL, SEED = 2.0, 8, 1.3356, 0.02, 0.995, 3000, 20261029
GRID = np.round(np.arange(0.10, 2.0001, 0.05), 2)                             # per-patch levels c
HBIN = np.linspace(0.0, 4.0, 401)                                               # histogram of d_p for c
CAL_RUN = "paired_record_ocl"


def patch_stats(Dp):
    """Dp (N_FUT, n_p) per-patch residuals of one decision (torch, outside the hand) -> per-draw q98, counts and excess
    mass above every GRID level, the number of patches, and the pooled histogram"""
    import torch
    g = torch.as_tensor(GRID, device=Dp.device, dtype=Dp.dtype)
    ex = (Dp[:, :, None] - g).clamp(min=0)                                      # (N_FUT, n_p, len(GRID))
    return {"q98": torch.quantile(Dp, 0.98, dim=1).cpu().numpy(), "cnt": (ex > 0).sum(1).cpu().numpy().astype(np.int16),
            "mass": ex.sum(1).cpu().numpy().astype(np.float32), "np": int(Dp.shape[1]),
            "hist": np.histogram(Dp.cpu().numpy().ravel(), HBIN)[0]}


def stat_value(st, kind, c):
    """per-draw score of one decision from patch_stats output -> (N_FUT,)"""
    if kind == "q98":
        return st["q98"]
    k = int(np.flatnonzero(np.isclose(GRID, c))[0])
    return (st["cnt"][:, k] if kind == "area" else st["mass"][:, k]) / st["np"]


def calib_select():
    """3000 seeded decisions of the calibration run (no outcome used)"""
    rows = []
    for f in sorted((ROOT / "results/jenga/hj2/data" / CAL_RUN).glob("ep*.npz")):
        with np.load(f) as d:
            k, g = d["chunk_index"], d["proprio"][:, 3]
        rows += [(CAL_RUN, f.stem, i, -1, int(k[i]), float(g[i] > 0)) for i in range(len(k))]
    rng = np.random.default_rng(SEED)
    return [rows[j] for j in sorted(rng.choice(len(rows), N_CAL, replace=False))]


def run(which):
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import cf_conf as CC
    import cf_fork as C
    import cf_rev as V
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0, total = cbff.CFG["PLAN_START"], 2 * wm.STEPS
    sel = (calib_select() if which == "calib" else CC.select())[i_sh::n_sh]; cache = {}
    res = {k: [] for k in ("run", "ep", "i", "kt", "k", "closed", "q98", "cnt", "mass", "np")}; hist = np.zeros(len(HBIN) - 1, np.int64)
    for n, (run_, stem, i, kt, k, cl) in enumerate(sel):
        if (run_, stem) not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz, hp = d["z"].astype(np.float32), d["proprio"]
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cache[(run_, stem)] = (hz, hp, plan16, target)
        hz, hp, plan16, target = cache[(run_, stem)]
        z0 = torch.as_tensor(hz[i], device="cuda")
        plan = np.asarray(plan16[i], np.float32)
        cur = np.r_[target[i], plan[0, 3]].astype(np.float32)
        seed = C.SEED + 1000 * int(stem[2:]) + i                                 # = CF3's / irrm1's draws for this decision
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        hold = np.repeat(cur[None], total, 0)
        rev = V.reverse_plan(plan, cur, NL, S0)
        rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
        mask = torch.as_tensor(~cb.hand_mask(np.vstack([hp[i, :3][None], plan[S0:S0 + NL, :3]])), device="cuda")
        acts = []
        for base in (rev, hold):
            for j in range(C.N_FUT):
                a = base.copy(); a[:NL, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:NL] * S; acts.append(a)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(2 * C.N_FUT, wm.STEPS, -1)
        z = st.repeat(2, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = model.step(z, pairs[:, t]).float()
        z = z.float().reshape(2, C.N_FUT, *z.shape[1:])
        ps = patch_stats((z[0] - z[1]).abs().mean(-1)[:, mask])
        hist += ps.pop("hist")
        for key, v in (("run", run_), ("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl)):
            res[key].append(v)
        for key, v in ps.items():
            res[key].append(v)
        if n % 100 == 0:
            print("rng1", which, i_sh, n, len(sel), flush=True)
    np.savez_compressed(OUT / f"rng1_{which}_{i_sh}.npz", hist=hist, **{k: np.asarray(v) for k, v in res.items()})
    print("saved", flush=True)


def load(which):
    fs = sorted(OUT.glob(f"rng1_{which}_*.npz"))
    D = [dict(np.load(f)) for f in fs]
    R = {k: np.concatenate([d[k] for d in D]) for k in D[0] if k != "hist"}
    R["hist"] = sum(d["hist"] for d in D)
    return R


def score(R, kind, c):
    if kind == "q98":
        return R["q98"].astype(np.float64).max(1)
    k = int(np.flatnonzero(np.isclose(GRID, c))[0])
    return ((R["cnt"][:, :, k] if kind == "area" else R["mass"][:, :, k]).astype(np.float64) / R["np"][:, None]).max(1)


def conformal(x, alpha):
    x = np.sort(x); k = min(len(x), ceil((len(x) + 1) * (1 - alpha)))
    return float(x[k - 1])


def grade():
    from sklearn.metrics import roc_auc_score
    Cal, Ev = load("calib"), load("cf3")
    I = {k: np.concatenate([np.load(f)[k] for f in sorted(OUT.glob("irrm1_rows_*.npz"))]) for k in ("run", "ep", "i", "r8")}
    import cf_conf as CC
    key = lambda run_, ep, i: (run_, int(ep), int(i))
    stored = {key(CC.RUNS[int(a)], b, c): r for a, b, c, r in zip(I["run"], I["ep"], I["i"], I["r8"])}
    dr = np.array([np.abs(Ev["q98"][n].astype(np.float64) - stored[key(Ev["run"][n], Ev["ep"][n], Ev["i"][n])]).max() for n in range(len(Ev["ep"]))])
    cdf = np.cumsum(Cal["hist"]) / Cal["hist"].sum()
    c_raw = float(HBIN[1:][np.searchsorted(cdf, Q_C)])
    c = float(GRID[min(len(GRID) - 1, np.searchsorted(GRID, c_raw - 1e-9))])
    kt, k, cl = Ev["kt"], Ev["k"], Ev["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0; ok = pre | safe
    g = lambda r, t: {"tau": round(t, 5), "recall_pre": round(float((r[pre] > t).mean()), 4),
                      "safe_closed": round(float((r[safe & cl] > t).mean()), 4), "safe_open": round(float((r[safe & ~cl] > t).mean()), 4)}
    ref = g(score(Ev, "q98", None), TAU0)
    out = {"reproduction": {"decisions": int(len(dr)), "max_abs_dr_vs_irrm1": float(dr.max()), "reproduced": bool(dr.max() <= 0.01)},
           "calibration": {"run": CAL_RUN, "decisions": int(len(Cal["ep"])), "c_pooled_q995": round(c_raw, 3), "c": c},
           "evaluation": {"decisions": int(len(kt)), "pre": int(pre.sum()), "safe_closed": int((safe & cl).sum()), "safe_open": int((safe & ~cl).sum())},
           "reference_current": ref | {"auc": round(float(roc_auc_score(pre[ok], score(Ev, "q98", None)[ok])), 4)}, "candidates": {}}
    for kind in ("q98", "area", "mass"):
        rc, re_ = score(Cal, kind, c), score(Ev, kind, c)
        tau = conformal(rc, ALPHA)
        sw = {m: g(re_, tau * m) for m in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)}
        o = {"tau": round(tau, 5), "auc": round(float(roc_auc_score(pre[ok], re_[ok])), 4), "at_tau": sw[1.0],
             "sweep": {str(m): v for m, v in sw.items()},
             "range_swing_pp": round(100 * (sw[0.85]["recall_pre"] - sw[1.15]["recall_pre"]), 1),
             "score_quantiles_eval": [round(float(x), 5) for x in np.quantile(re_, [0.5, 0.9, 0.95, 0.99, 1.0])],
             "pre_median_over_tau": round(float(np.median(re_[pre]) / tau), 3) if tau > 0 else None}
        if kind != "q98":                                                       # q98 recalibrated: reported only
            o["pass"] = {"range": o["range_swing_pp"] <= 20, "auc": o["auc"] >= 0.75, "safe_closed": o["at_tau"]["safe_closed"] <= 0.02,
                         "safe_open": o["at_tau"]["safe_open"] <= 0.10, "recall": o["at_tau"]["recall_pre"] >= ref["recall_pre"]}
            o["R_pass"] = bool(all(o["pass"].values()))
        out["candidates"][kind] = o
    P = [kd for kd in ("area", "mass") if out["candidates"][kd]["R_pass"]]
    P.sort(key=lambda kd: (out["candidates"][kd]["at_tau"]["recall_pre"], out["candidates"][kd]["auc"]), reverse=True)
    out["selected"] = {"stat": P[0], "c": c, "tau": out["candidates"][P[0]]["tau"]} if P else None
    out["verdict"] = f"PASS: {P[0]} selected" if P else "STOP: no candidate passes"
    (OUT / "rng1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"calib": lambda: run("calib"), "cf3": lambda: run("cf3"), "grade": grade}[sys.argv[1]]()
