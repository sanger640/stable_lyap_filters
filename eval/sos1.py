"""SOS1 stage S (results/jenga/sos1_protocol.json, offline): a small-object undo score, so red-block falls become visible.
Why: the undo-test score r = max_j q98_outside-hand d_p (d_p = mean over the 8 latent channels of |f_rev_j - f_hold_j| per
patch) fires only when ~12 of the ~600 outside-hand patches change; a fallen red block changes ~7 (REDM1, NM1, G0: pre-red
recall 0-5%). G0's single-patch max reaches 19-38% but flags 6-13% of clean grasping moves. Candidates in between, per draw j
on the 28 x 28 patch grid, ALL patches (amendment 1, 2026-11-01, user: no hand mask):
  q98   reference (= irrm1 r8 at NL 8, reproduced)          q995  99.5th percentile (~3 patches)
  top5  mean of the 5 largest patches                       pool2 max over 2 x 2 windows of the mean d_p (adjacent change)
  pool3 max over 3 x 3 windows of the mean d_p              max   single largest patch (reported only)
score = max over the 64 draws; NSI (irreversible fragility, reported only) = score > tau AND the median draw <= tau.
tau per score and horizon = conformal 98% over the calibration decisions (policy's own moves, label-free, as RNG1).
Calibration: RNG1's 3000 seeded decisions of paired_record_ocl. Evaluation: CF3's 4679 fresh decisions (labels grade only;
pre-red = 1-3 chunks before a red-block fall, redm1_rows; pre-topple = 1-3 chunks before a neighbour topple).
Horizons NL 8 (0.8 s, the confirmed value's labels) and 16. Gripper held at the current command in the undo branch (as the
value's labels, eval/irrm2.py); noise, seeds, mask and hold branch exactly as irrm1 / RNG1.
usage: SHARD=i/n python3 eval/sos1.py calib|cf3 -> results/jenga/sos1_{calib,cf3}_<i>.npz;  python3 eval/sos1.py grade
       -> results/jenga/sos1_result.json   (SOS1_START / SOS1_LIMIT: a slice of the shard, smoke test only)"""
import json
import os
from math import ceil
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga"
S, NLS, TAU0, ALPHA = 2.0, (8, 16), 1.3356, 0.02
STATS = ("q98", "q995", "top5", "pool2", "pool3", "max")
CANDS = ("q995", "top5", "pool2", "pool3")


def patch_scores(D, mask):
    """D (N_FUT, 28, 28) per-patch residuals, mask (28, 28) bool outside the hand -> {stat: (N_FUT,) numpy}"""
    import torch
    import torch.nn.functional as F
    x = D[:, mask]
    Dm = (D * mask).unsqueeze(1)
    return {"q98": torch.quantile(x, 0.98, dim=1), "q995": torch.quantile(x, 0.995, dim=1),
            "top5": x.topk(5, dim=1).values.mean(1), "max": x.max(1).values,
            "pool2": F.avg_pool2d(Dm, 2, stride=1).flatten(1).max(1).values,
            "pool3": F.avg_pool2d(Dm, 3, stride=1).flatten(1).max(1).values}


def run(which):
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import cf_conf as CC
    import cf_fork as C
    import cf_rev as V
    import rng1
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0, total = cbff.CFG["PLAN_START"], 2 * wm.STEPS
    sel = (rng1.calib_select() if which == "calib" else CC.select())[i_sh::n_sh]
    sel = sel[int(os.environ.get("SOS1_START", 0)):][:int(os.environ.get("SOS1_LIMIT", len(sel)))]
    res = {k: [] for k in ("run", "ep", "i", "kt", "k", "closed")}; sc = {f"{s}{nl}": [] for nl in NLS for s in STATS}; cache = {}
    for n, (run_, stem, i, kt, k, cl) in enumerate(sel):
        if (run_, stem) not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz, hp = d["z"].astype(np.float32), d["proprio"]
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cache[(run_, stem)] = (hz, hp, plan16, target)
        hz, hp, plan16, target = cache[(run_, stem)]
        z0 = torch.as_tensor(hz[i], device="cuda"); plan = np.asarray(plan16[i], np.float32)
        cur = np.r_[target[i], plan[0, 3]].astype(np.float32)
        seed = C.SEED + 1000 * int(stem[2:]) + i                                 # = CF3's / irrm1's / RNG1's draws
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        hold = np.repeat(cur[None], total, 0)
        acts, masks = [], []
        for nl in NLS:
            rev = V.reverse_plan(plan, cur, nl, S0)
            rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
            rev[:, 3] = cur[3]                                                   # gripper held, as the value's labels
            masks.append(torch.ones(28, 28, dtype=torch.bool, device="cuda"))     # amendment 1: no hand mask (all patches)
            for base in (rev, hold):
                for j in range(C.N_FUT):
                    a = base.copy(); a[:nl, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:nl] * S; acts.append(a)
        B = 2 * len(NLS)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(B * C.N_FUT, wm.STEPS, -1)
        z = st.repeat(B, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = model.step(z, pairs[:, t]).float()
        z = z.float().reshape(B, C.N_FUT, *z.shape[1:])
        for b, nl in enumerate(NLS):
            D = (z[2 * b] - z[2 * b + 1]).abs().mean(-1)                         # (N_FUT, 28, 28)
            for s, v in patch_scores(D, masks[b]).items():
                sc[f"{s}{nl}"].append(v.cpu().numpy().astype(np.float32))
        for key, v in (("run", run_), ("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl)):
            res[key].append(v)
        if n % 100 == 0:
            print("sos1", which, i_sh, n, len(sel), flush=True)
    np.savez_compressed(OUT / f"sos1_{which}_{i_sh}.npz", **{k: np.asarray(v) for k, v in res.items()},
                        **{k: np.asarray(v) for k, v in sc.items()})
    print("saved", flush=True)


def load(which):
    D = [dict(np.load(f)) for f in sorted(OUT.glob(f"sos1_{which}_*.npz"))]
    return {k: np.concatenate([d[k] for d in D]) for k in D[0]}


def conformal(x, alpha):
    x = np.sort(x); k = min(len(x), ceil((len(x) + 1) * (1 - alpha)))
    return float(x[k - 1])


def grade():
    from sklearn.metrics import roc_auc_score
    import cf_conf as CC
    Cal, Ev = load("calib"), load("cf3")
    I = {k: np.concatenate([np.load(f)[k] for f in sorted(OUT.glob("irrm1_rows_*.npz"))]) for k in ("run", "ep", "i", "r8")}
    key = lambda run_, ep, i: (str(run_), int(ep), int(i))
    stored = {key(CC.RUNS[int(a)], b, c): r for a, b, c, r in zip(I["run"], I["ep"], I["i"], I["r8"])}
    Q = np.load(OUT / "redm1_rows.npz")
    kr_of = {(CC.RUNS[int(a)], int(b), int(c)): int(d) for a, b, c, d in zip(Q["run"], Q["ep"], Q["k"], Q["kr"])}
    kt, k, cl = Ev["kt"], Ev["k"], Ev["closed"] > 0
    kr = np.array([kr_of.get((str(a), int(b), int(c)), -1) for a, b, c in zip(Ev["run"], Ev["ep"], Ev["k"])])
    pre_red = (kr >= 0) & (kr - k >= 1) & (kr - k <= 3); pre_top = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3) & ~pre_red
    clean = (kt < 0) & (kr < 0)
    # reproduction: q98 with the gripper held, all patches = G0's held{8,16}_all_q98 per draw on G0's 1657 decisions (must be exact);
    # vs irrm1's r8 (gripper with the plan) agreement is reported only
    import gfx1_grip as G0
    gi = np.concatenate([np.load(f)["idx"] for f in sorted(OUT.glob("gfx1_grip_rows_*.npz"))])
    gr = np.concatenate([np.load(f)["r"] for f in sorted(OUT.glob("gfx1_grip_rows_*.npz"))])
    pos = {key(CC.RUNS[int(I["run"][x])], I["ep"][x], I["i"][x]): n for n, x in enumerate(gi)}
    rep = {}
    for nl in NLS:
        v = G0.VARS.index(f"held{nl}_all_q98")
        d0 = [np.abs(Ev[f"q98{nl}"][n].astype(np.float64) - gr[pos[key(Ev["run"][n], Ev["ep"][n], Ev["i"][n])], v]).max()
              for n in range(len(kt)) if key(Ev["run"][n], Ev["ep"][n], Ev["i"][n]) in pos]
        rep[f"NL{nl}"] = {"decisions": len(d0), "max_abs": round(float(np.max(d0)), 5), "reproduced": bool(np.max(d0) <= 0.01)}
    dr = np.array([np.abs(Ev["q988"][n].astype(np.float64) - stored[key(Ev["run"][n], Ev["ep"][n], Ev["i"][n])]).max()
                   for n in range(len(kt))])
    auc = lambda s, pos: round(float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(clean.sum())], np.r_[s[pos], s[clean]])), 4)
    out = {"protocol": "results/jenga/sos1_protocol.json",
           "evaluation": {"decisions": int(len(kt)), "pre_red": int(pre_red.sum()), "pre_topple": int(pre_top.sum()),
                          "clean_closed": int((clean & cl).sum()), "clean_open": int((clean & ~cl).sum())},
           "reproduction_vs_G0_held": rep, "reproduced": bool(all(v["reproduced"] for v in rep.values())),
           "q98_vs_irrm1_r8_plan_gripper": {"max_abs": round(float(dr.max()), 5), "median_abs": round(float(np.median(dr)), 5),
                               "share_exact": round(float((dr <= 0.01).mean()), 4)},
           "calibration_decisions": int(len(Cal["ep"])), "scores": {}}
    for nl in NLS:
        for s in STATS:
            name = f"{s}{nl}"; rc, re_ = Cal[name].astype(np.float64), Ev[name].astype(np.float64)
            tau = conformal(rc.max(1), ALPHA); r = re_.max(1); med = np.median(re_, 1)
            at = lambda f, t=tau: {"pre_red": round(float((f[pre_red] > t).mean()), 4), "pre_topple": round(float((f[pre_top] > t).mean()), 4),
                                   "clean_closed": round(float((f[clean & cl] > t).mean()), 4), "clean_open": round(float((f[clean & ~cl] > t).mean()), 4)}
            nsi = ((r > tau) & (med <= tau)).astype(float)
            t2 = float(np.quantile(r[clean & cl], 0.98))                         # matched false alarm, reported only
            o = {"tau": round(tau, 5), "auc_pre_red": auc(r, pre_red), "auc_pre_topple": auc(r, pre_top), "at_tau": at(r),
                 "nsi_at_tau": at(nsi, 0.5), "matched_2pct_closed": at(r, t2) | {"thr": round(t2, 5)}}
            if s in CANDS:
                o["pass"] = {"clean_closed": o["at_tau"]["clean_closed"] <= 0.02, "clean_open": o["at_tau"]["clean_open"] <= 0.03,
                             "pre_red": o["at_tau"]["pre_red"] >= 0.20}
                o["S_pass"] = bool(all(o["pass"].values()))
            out["scores"][name] = o
    for nl in NLS:                                                               # topple recall must not drop vs q98 at the same horizon
        ref = out["scores"][f"q98{nl}"]["at_tau"]["pre_topple"]
        for s in CANDS:
            o = out["scores"][f"{s}{nl}"]; o["pass"]["pre_topple"] = o["at_tau"]["pre_topple"] >= ref - 0.02
            o["S_pass"] = bool(all(o["pass"].values()))
    P = [n for n in out["scores"] if out["scores"][n].get("S_pass")]
    rank = lambda n: (n.endswith("16") is False, out["scores"][n]["at_tau"]["pre_red"], out["scores"][n]["at_tau"]["pre_topple"])
    P.sort(key=rank, reverse=True)
    out["selected"] = {"score": P[0], "tau": out["scores"][P[0]]["tau"]} if P else None
    out["verdict"] = f"PASS: {P[0]} selected" if P else "STOP: no candidate passes"
    (OUT / "sos1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"calib": lambda: run("calib"), "cf3": lambda: run("cf3"), "grade": grade}[sys.argv[1]]()
