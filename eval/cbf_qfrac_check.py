"""Imagined topple fraction q: fraction of futures in the 2-group split's group with larger change away from the hand.
Does q (from the world model) track real risk across plans of a decision, and do nudge-induced changes in q track real
risk changes? Compared for the old and the fine-tuned model."""
import json, os, sys, hashlib
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb, step3_world_model as wm
C, K = q.load_cache()
T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
res = {}
for name, path in (("old", "results/program_b/closedloop_v1/model/model_US.pt"), ("new", "results/program_b/cf_v1/model/model_CF.pt")):
    os.environ["CBF_WM"] = os.path.abspath(path)
    mon = cf.Monitor()
    sp, dq, dr, pick = [], [], [], []
    for r in rows:
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        P = np.asarray(r["plans"], np.float32)
        risk = np.array([T[q._hash(p[1:9])] for p in P])
        bar = q.Barrier(mon, z0, 1000 + i, 64, 1.0)
        with torch.no_grad():
            _, e30 = bar.endpoints(torch.as_tensor(P, device="cuda"))
            _, lab, _ = cf.split_test_gpu(e30)
        qs = []
        for c in range(len(P)):
            pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            diff = (e30[c].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1)
            o_j = (diff * m).sum((1, 2)) / m.sum()
            L = lab[c].bool()
            if L.all() or (~L).all():
                qs.append(0.0); continue
            bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
            qs.append(float(bad.float().mean()))
        qs = np.array(qs)
        S = slice(0, 13)
        if np.ptp(risk[S]) > 0 and np.ptp(qs[S]) > 0:
            sp.append(spearmanr(-qs[S], -risk[S]).correlation)
        pick.append(risk[S][np.argmin(qs[S])])
        dq += list(qs[13:] - qs[0]); dr += list(risk[13:] - risk[0])
    res[name] = {"spearman_q_vs_risk": round(float(np.nanmean(sp)), 3), "risk_pick_min_q": round(float(np.mean(pick)), 4),
                 "nudge_dq_vs_dr_corr": round(float(np.corrcoef(dq, dr)[0, 1]), 3)}
    print(name, res[name], flush=True)
json.dump(res, open(q.OUT / "qfrac_check.json", "w"), indent=1)
