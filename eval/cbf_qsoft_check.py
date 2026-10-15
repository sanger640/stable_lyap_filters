"""Smooth risk-like barriers from the imagined futures (differentiable, no split needed), new vs old model:
mean_o = mean over futures of the change away from the hand; q_soft(T) = mean_j sigmoid((o_j - O_MAX) / (T * O_MAX)).
Same 75 risky decisions / nudges / samples as cbf_qp_checks (truth cached)."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb, cbf_filter as cbff
C, K = q.load_cache()
T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
O_MAX = json.loads((q.ROOT / "results/jenga/cbf_calibration.json").read_text())["O_MAX"]
TEMPS = (0.05, 0.1, 0.2)
res = {}
for name, path in (("old", "results/program_b/closedloop_v1/model/model_US.pt"), ("new", "results/program_b/cf_v1/model/model_CF.pt")):
    os.environ["CBF_WM"] = os.path.abspath(path)
    mon = cf.Monitor()
    stats = {k: {"dq": [], "sp": [], "pick": []} for k in ["mean_o"] + [f"q_soft_{t}" for t in TEMPS]}
    dr_all = []
    for r in rows:
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        P = np.asarray(r["plans"], np.float32)
        risk = np.array([T[q._hash(p[1:9])] for p in P])
        bar = q.Barrier(mon, z0, 1000 + i, 64, 1.0)
        with torch.no_grad():
            _, e30 = bar.endpoints(torch.as_tensor(P, device="cuda"))
        O = []
        for c in range(len(P)):
            pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            diff = (e30[c].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1)
            O.append(((diff * m).sum((1, 2)) / m.sum()).cpu().numpy())
        O = np.stack(O)                                                       # (plans, 64)
        vals = {"mean_o": O.mean(1)}
        for t in TEMPS:
            vals[f"q_soft_{t}"] = (1 / (1 + np.exp(-(O - O_MAX) / (t * O_MAX)))).mean(1)
        S = slice(0, 13)
        for k, v in vals.items():
            if np.ptp(risk[S]) > 0 and np.ptp(v[S]) > 0:
                stats[k]["sp"].append(spearmanr(v[S], risk[S]).correlation)
            stats[k]["pick"].append(risk[S][np.argmin(v[S])])
            stats[k]["dq"] += list(v[13:] - v[0])
        dr_all += list(risk[13:] - risk[0])
    res[name] = {k: {"nudge_corr": round(float(np.corrcoef(s["dq"], dr_all)[0, 1]), 3), "spearman_samples": round(float(np.nanmean(s["sp"])), 3),
                     "risk_pick_min": round(float(np.mean(s["pick"])), 4)} for k, s in stats.items()}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(q.OUT / "qsoft_check.json", "w"), indent=1)
