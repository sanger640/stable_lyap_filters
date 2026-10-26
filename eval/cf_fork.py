"""Counterfactual fork statistic, offline (results/jenga/cf1_protocol.json). The intuitive-physics-engine fork unchanged
(64 imagined futures in model_OIL under perception + execution noise at scale s, as eval/cbf_filter.py), but read
counterfactually: the SAME 64 noise draws are rolled out under the policy's plan and under holding still, and per draw j
  d_j = 98th percentile over the latent patches OUTSIDE the plan's hand path of mean_channels |f_plan_j - f_hold_j|
at the 30-step end (what the plan changes in the scene beyond what it deliberately moves, relative to what would have
happened anyway; background motion common to both branches cancels). P_cf = share of draws with d_j > tau, tau = the
98th percentile of d_j over pre-grasp decisions (label-free: proprio only). Also, from the same rollouts: the hold
branch's spread (the existing fork statistic) and o (plan vs the current latent outside the hand path, the fork CBF's
regime-change term). Inputs: held-out m3 decisions (z from results/jenga/hj2/data, plans from the rollout npz); no
DINOv2. Labels for grading only.
usage: python3 eval/cf_fork.py run   -> results/jenga/cf1_rows.npz;  python3 eval/cf_fork.py grade -> results/jenga/cf1_result.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
HJ = ROOT / "results/jenga/hj2/data/paired_record_m3"
PR = ROOT / "results/program_b/policy_rollouts/paired_record_m3"
OUT = ROOT / "results/jenga"
SCALES, N_FUT, SEED = (2.0, 4.0), 64, 20261026


def select():
    """All pre-topple decisions (1-3 chunks ahead), all safe-episode gripper-closed decisions, 1200 other safe-episode
    decisions (seeded); grading labels are only used to pick the sample, never in the statistic."""
    rows = []
    for f in sorted(HJ.glob("ep*.npz")):
        with np.load(f) as d:
            kt, k, g = int(d["topple_chunk"]), d["chunk_index"], d["proprio"][:, 3]
        for i in range(len(k)):
            rows.append((f.stem, i, kt, int(k[i]), float(g[i] > 0)))
    rng = np.random.default_rng(SEED)
    pre = [r for r in rows if r[2] >= 0 and 1 <= r[2] - r[3] <= 3]
    sc = [r for r in rows if r[2] < 0 and r[4] > 0]
    so = [r for r in rows if r[2] < 0 and r[4] <= 0]
    so = [so[i] for i in sorted(rng.choice(len(so), 1200, replace=False))]
    return pre + sc + so


def run():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    sel = select(); out = OUT / "cf1_rows.npz"
    cache = {}
    res = {k: [] for k in ("ep", "i", "kt", "k", "closed", "pregrasp")}
    for s in SCALES:
        for q in ("d", "o", "e"):
            res[f"{q}_{s:g}"] = []
        res[f"spread_{s:g}"] = []
    for n, (stem, i, kt, k, cl) in enumerate(sel):
        if stem not in cache:
            cache.clear()
            with np.load(HJ / f"{stem}.npz") as d:
                hz, hp = d["z"].astype(np.float32), d["proprio"]
            with np.load(PR / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cl_idx = np.flatnonzero(hp[:, 3] > 0); fc = cl_idx[0] if len(cl_idx) else len(hp)
            cache[stem] = (hz, hp, plan16, target, fc)
        hz, hp, plan16, target, fc = cache[stem]
        z0 = torch.as_tensor(hz[i], device="cuda")
        plan = np.asarray(plan16[i], np.float32)
        hold = np.repeat(np.r_[target[i], plan[0, 3]][None].astype(np.float32), 16, 0)
        pts = np.vstack([hp[i, :3][None], plan[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + cbff.CFG["N_LOOK"], :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
        for key, v in (("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl), ("pregrasp", float(i < fc))):
            res[key].append(v)
        seed = SEED + 1000 * int(stem[2:]) + i
        for s in SCALES:
            d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                           for p, r in (E.sample(seed, 0, 0, j) for j in range(N_FUT))])
            st = G(z0[None].expand(N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
            acts = []
            for p in (plan, hold):
                base = cbff.horizon(p, cbff.CFG["N_LOOK"])
                for j in range(N_FUT):
                    a = base.copy()
                    a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
                    acts.append(a)
            pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(2 * N_FUT, wm.STEPS, -1)
            z = st.repeat(2, 1, 1, 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(wm.STEPS):
                    z = model.step(z, pairs[:, t]).float()
                    if t == wm.HOLD30:
                        break
            zp, zh = z[:N_FUT].float(), z[N_FUT:].float()                     # (64, 28, 28, 8)
            dm = (zp - zh).abs().mean(-1)[:, m]                                  # (64, n_outside)
            om = (zp - z0[None]).abs().mean(-1)[:, m]
            em = (zh - z0[None]).abs().mean(-1)[:, m]
            res[f"d_{s:g}"].append(torch.quantile(dm, 0.98, dim=1).cpu().numpy())
            res[f"o_{s:g}"].append(torch.quantile(om, 0.98, dim=1).cpu().numpy())
            res[f"e_{s:g}"].append(torch.quantile(em, 0.98, dim=1).cpu().numpy())
            x = zh.reshape(N_FUT, -1).double(); X = x - x.mean(0, keepdim=True)
            w = torch.linalg.eigvalsh(X @ X.T)
            res[f"spread_{s:g}"].append(float(w[-1] / N_FUT / x.shape[1]))
        if n % 100 == 0:
            print("cf1", n, len(sel), flush=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in res.items()})
    print("saved", out, flush=True)


def grade():
    from sklearn.metrics import roc_auc_score
    R = dict(np.load(OUT / "cf1_rows.npz"))
    kt, k, cl, pg = R["kt"], R["k"], R["closed"] > 0, R["pregrasp"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0
    out = {"decisions": int(len(kt)), "pre": int(pre.sum()), "safe": int(safe.sum()), "safe_closed": int((safe & cl).sum()),
           "pregrasp_reference": int(pg.sum()), "stats": {}}
    for s in SCALES:
        for q in ("d", "o", "e"):
            D = R[f"{q}_{s:g}"]                                                 # (n, 64)
            tau = float(np.quantile(D[pg].ravel(), 0.98))
            for name, r in ((f"P_{q}_{s:g}", (D > tau).mean(1)), (f"mean_{q}_{s:g}", D.mean(1)), (f"max_{q}_{s:g}", D.max(1))):
                out["stats"][name] = _grade(r, pre, safe, cl)
            out["stats"][f"P_{q}_{s:g}"]["tau"] = tau
        out["stats"][f"spread_{s:g}"] = _grade(np.log(R[f"spread_{s:g}"] + 1e-12), pre, safe, cl)
    best_ref = min(out["stats"][f"spread_{s:g}"]["safe_closed_flag_at_50pct"] for s in SCALES)
    cf = {n: v for n, v in out["stats"].items() if n.startswith("P_d_")}
    out["reference_spread_safe_closed_flag_at_50pct"] = best_ref
    out["GATE_CF1_pass"] = bool(any(v["auc"] >= 0.85 and v["safe_closed_flag_at_50pct"] <= best_ref / 2 for v in cf.values()))
    (OUT / "cf1_result.json").write_text(json.dumps(out, indent=1))
    for n, v in sorted(out["stats"].items(), key=lambda x: x[1]["safe_closed_flag_at_50pct"]):
        print(f"{n:14s} AUC {v['auc']:.3f}  @50% recall: safe {v['safe_flag_at_50pct']:.3f} safe&closed {v['safe_closed_flag_at_50pct']:.3f}")
    print("GATE_CF1_pass", out["GATE_CF1_pass"])


def _grade(r, pre, safe, cl):
    from sklearn.metrics import roc_auc_score
    m = pre | safe
    thr = np.quantile(r[pre], 0.5)
    return {"auc": round(float(roc_auc_score(pre[m], r[m])), 4), "safe_flag_at_50pct": round(float(np.mean(r[safe] >= thr)), 4),
            "safe_closed_flag_at_50pct": round(float(np.mean(r[safe & cl] >= thr)), 4)}


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()
