"""CF3: confirmatory test of the fork's worst-case irreversibility (results/jenga/cf3_protocol.json). The statistic is
CF2's (eval/cf_rev.py) at 2x noise, read as the max over the 64 draws: r_max = max_j q98_outside-hand |f_rev_j - f_hold_j|,
flagged iff r_max > THR = 1.3356 (LF6's label-free scale: q98 of real latent change before the first grasp, training
runs). Fresh decisions from paired_record_s4 + paired_record_h2h2 (not used to choose any CF statistic), selected as CF1.
Also the hold branch's spread at the same end (reference). Labels for grading only.
usage: SHARD=i/n python3 eval/cf_conf.py run -> results/jenga/cf3_rows_<i>.npz;  python3 eval/cf_conf.py grade -> cf3_result.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cf_fork as C
import cf_rev as V

OUT = ROOT / "results/jenga"
RUNS = ("paired_record_s4", "paired_record_h2h2")
S, THR = 2.0, 1.3356


def select():
    rows = []
    for run in RUNS:
        C.HJ = ROOT / "results/jenga/hj2/data" / run
        rows += [(run,) + r for r in C.select()]
    return rows


def run():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    S0, NL = cbff.CFG["PLAN_START"], cbff.CFG["N_LOOK"]
    total = 2 * wm.STEPS
    sel = select()[i_sh::n_sh]; cache = {}
    res = {k: [] for k in ("run", "ep", "i", "kt", "k", "closed", "r", "spread")}
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
        rev = V.reverse_plan(plan, cur, NL, S0)
        rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
        hold = np.repeat(cur[None], total, 0)
        pts = np.vstack([hp[i, :3][None], plan[S0:S0 + NL, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
        for key, v in (("run", RUNS.index(run_)), ("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl)):
            res[key].append(v)
        seed = C.SEED + 1000 * int(stem[2:]) + i
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        acts = []
        for base in (rev, hold):
            for j in range(C.N_FUT):
                a = base.copy()
                a[:NL, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:NL] * S
                acts.append(a)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(2 * C.N_FUT, wm.STEPS, -1)
        z = st.repeat(2, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = model.step(z, pairs[:, t]).float()
        zr, zh = z[:C.N_FUT].float(), z[C.N_FUT:].float()
        res["r"].append(torch.quantile((zr - zh).abs().mean(-1)[:, m], 0.98, dim=1).cpu().numpy())
        x = zh.reshape(C.N_FUT, -1).double(); X = x - x.mean(0, keepdim=True)
        res["spread"].append(float(torch.linalg.eigvalsh(X @ X.T)[-1] / C.N_FUT / x.shape[1]))
        if n % 100 == 0:
            print("cf3", i_sh, n, len(sel), flush=True)
    np.savez_compressed(OUT / f"cf3_rows_{i_sh}.npz", **{k: np.asarray(v) for k, v in res.items()})
    print("saved", flush=True)


def grade():
    from sklearn.metrics import roc_auc_score
    fs = sorted(OUT.glob("cf3_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
    kt, k, cl = R["kt"], R["k"], R["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0
    m = R["r"].max(1); ok = pre | safe
    fl = m > THR
    out = {"decisions": int(len(kt)), "pre": int(pre.sum()), "safe_closed": int((safe & cl).sum()), "safe_open": int((safe & ~cl).sum()),
           "auc_max_r": round(float(roc_auc_score(pre[ok], m[ok])), 4),
           "at_thr": {"thr": THR, "recall_pre": round(float(fl[pre].mean()), 4), "safe_closed": round(float(fl[safe & cl].mean()), 4),
                      "safe_open": round(float(fl[safe & ~cl].mean()), 4),
                      "recall_by_lead": [round(float(fl[pre & (kt - k == d)].mean()), 4) for d in (1, 2, 3)]},
           "at_50pct_recall": {"max_r": C._grade(m, pre, safe, cl), "hold_spread": C._grade(np.log(R["spread"] + 1e-12), pre, safe, cl)},
           "per_run": {}}
    for j, run_ in enumerate(RUNS):
        w = R["run"] == j
        out["per_run"][run_] = {"auc": round(float(roc_auc_score(pre[ok & w], m[ok & w])), 4), "recall_pre": round(float(fl[pre & w].mean()), 4),
                                "safe_closed": round(float(fl[safe & cl & w].mean()), 4), "safe_open": round(float(fl[safe & ~cl & w].mean()), 4)}
    a = out["at_thr"]
    out["GATE_CF3_pass"] = bool(out["auc_max_r"] >= 0.85 and a["recall_pre"] >= 0.40 and a["safe_closed"] <= 0.05 and a["safe_open"] <= 0.10)
    (OUT / "cf3_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()
