"""IRRM1 (results/jenga/irrm_protocol.json, stage 1, offline): which failure margin for the always-on edit's value?
On CF3's fresh decisions (eval/cf_conf.py select(): paired_record_s4 + paired_record_h2h2, labels for grading only), score
  irreversibility at a short action horizon: CF3's undo test (eval/cf_conf.py) with the plan's first NL control steps
    instead of 15 (NL = 2: one model step, 0.2 s; NL = 8: one move, 0.8 s): do them, undo back to the current target,
    hold; vs hold; 64 draws at 2x noise; r_max = max_j q98_outside-hand |f_rev_j - f_hold_j|; failure iff r_max > 1.3356
  fragility: the current value's margin net l_hat(z) (results/program_b/lsf_v1/margin_lf.pt); failure iff l_hat < 0
  union: either
and grade each against pre-topple (1-3 chunks ahead) vs safe-episode decisions (gripper closed / open).
usage: SHARD=i/n python3 eval/irrm1.py run -> results/jenga/irrm1_rows_<i>.npz;  python3 eval/irrm1.py grade -> irrm1_result.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cf_conf as CC
import cf_fork as C
import cf_rev as V

OUT = ROOT / "results/jenga"
S, THR, NLS = 2.0, 1.3356, (2, 8)


def run():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import step3b_models as sm
    import step3_world_model as wm
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    os.environ["LSF_CLF"] = "margin_lf.pt"
    import lsf as L
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
    G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
    clf = L.load_clf("cuda")
    S0 = cbff.CFG["PLAN_START"]
    total = 2 * wm.STEPS
    sel = CC.select()[i_sh::n_sh]; cache = {}
    res = {k: [] for k in ("run", "ep", "i", "kt", "k", "closed", "frag")}
    res.update({f"r{nl}": [] for nl in NLS})
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
        for key, v in (("run", CC.RUNS.index(run_)), ("ep", int(stem[2:])), ("i", i), ("kt", kt), ("k", k), ("closed", cl)):
            res[key].append(v)
        with torch.no_grad():
            res["frag"].append(float(clf(z0[None].float(), torch.zeros(1, 4, device="cuda"))))
        seed = C.SEED + 1000 * int(stem[2:]) + i                       # = CF3's draws for this decision
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
        st = G(z0[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        hold = np.repeat(cur[None], total, 0)
        acts, masks = [], []
        for nl in NLS:                                                 # per horizon: do-then-undo and hold, same noise (as CF3)
            rev = V.reverse_plan(plan, cur, nl, S0)
            rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
            masks.append(torch.as_tensor(~cb.hand_mask(np.vstack([hp[i, :3][None], plan[S0:S0 + nl, :3]])), device="cuda"))
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
            res[f"r{nl}"].append(torch.quantile((z[2 * b] - z[2 * b + 1]).abs().mean(-1)[:, masks[b]], 0.98, dim=1).cpu().numpy())
        if n % 100 == 0:
            print("irrm1", i_sh, n, len(sel), flush=True)
    np.savez_compressed(OUT / f"irrm1_rows_{i_sh}.npz", **{k: np.asarray(v) for k, v in res.items()})
    print("saved", flush=True)


def grade():
    from sklearn.metrics import roc_auc_score
    fs = sorted(OUT.glob("irrm1_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
    kt, k, cl = R["kt"], R["k"], R["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0; ok = pre | safe
    cands = {"fragility": (-R["frag"], R["frag"] < 0)}
    for nl in NLS:
        m = R[f"r{nl}"].max(1)
        cands[f"irr_{nl}"] = (m, m > THR)
        cands[f"union_{nl}"] = (np.maximum(m / THR, 1 - R["frag"]), (m > THR) | (R["frag"] < 0))
    out = {"decisions": int(len(kt)), "pre": int(pre.sum()), "safe_closed": int((safe & cl).sum()), "safe_open": int((safe & ~cl).sum()),
           "reference_cf3_irr_15": "results/jenga/cf3_result.json (recall 49.9%, safe closed 2.2%, safe open 7.6%, AUC 0.865)", "candidates": {}}
    for name, (score, fl) in cands.items():
        out["candidates"][name] = {"auc": round(float(roc_auc_score(pre[ok], score[ok])), 4),
                                   "recall_pre": round(float(fl[pre].mean()), 4), "safe_closed": round(float(fl[safe & cl].mean()), 4),
                                   "safe_open": round(float(fl[safe & ~cl].mean()), 4),
                                   "recall_by_lead": [round(float(fl[pre & (kt - k == d)].mean()), 4) for d in (1, 2, 3)]}
    (OUT / "irrm1_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()
