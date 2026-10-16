"""Readout fix: 'no fork' must not mean 'safe' when the single outcome is disturbed (everything topples).
q_fix = fork ? (share of futures in the disturbed group) : (1 if the futures' mean change away from the hand > tau else 0)
q_thr = share of futures whose change away from the hand exceeds tau (no split test at all).
tau = label-free: O_MAX (99th percentile of the policy's own max outside-hand change; results/jenga/cbf_calibration.json)
times a factor. Evaluated on REAL futures (rank_ceiling latents) = the readout's ceiling."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cbf_barrier as cb, cem_filter as cf
C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
O_MAX = json.loads((q.ROOT / "results/jenga/cbf_calibration.json").read_text())["O_MAX"]
OJ, FK, LAB, R = [], [], [], []
for r in rows:
    i = r["i"]; f = np.load(q.OUT / "rank_ceiling" / f"d{i}.npz")
    z = torch.as_tensor(f["z"].astype(np.float32), device="cuda"); z0 = torch.as_tensor(f["z0"].astype(np.float32), device="cuda")
    P = np.asarray(r["plans"], np.float32)[:13]
    R.append([T[q._hash(p[1:9])] for p in P])
    ok10, _, _ = cf.split_test_gpu(z[:, :, 0].reshape(13, 32, -1)); ok30, lab, _ = cf.split_test_gpu(z[:, :, 1].reshape(13, 32, -1))
    oj = []
    for c in range(13):
        pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        oj.append((((z[c, :, 1] - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()).cpu().numpy())
    OJ.append(np.stack(oj)); FK.append((ok10 & ok30).cpu().numpy()); LAB.append(lab.cpu().numpy())
OJ, FK, LAB, R = np.stack(OJ), np.stack(FK), np.stack(LAB), np.array(R)          # (75,13,32), (75,13), (75,13,32), (75,13)
def ev(S):
    sp = [spearmanr(S[d], R[d]).correlation for d in range(len(R)) if np.ptp(S[d]) > 0 and np.ptp(R[d]) > 0]
    pick = R[np.arange(len(R)), np.argmin(S + 1e-6 * np.arange(13), 1)]
    return {"spearman": round(float(np.mean(sp)), 3), "pick_real_risk": round(float(pick.mean()), 4)}
res = {"oracle": round(float(R.min(1).mean()), 4), "random": round(float(R[:, 1:].mean()), 4), "policy": round(float(R[:, 0].mean()), 4)}
for fac in (0.25, 0.5, 1.0):
    tau = fac * O_MAX
    q_thr = (OJ > tau).mean(-1)
    q_fix = np.zeros_like(q_thr)
    for d in range(len(R)):
        for c in range(13):
            L = LAB[d, c].astype(bool); o = OJ[d, c]
            if FK[d, c] and not (L.all() or (~L).all()):
                bad = L if o[L].mean() > o[~L].mean() else ~L
                q_fix[d, c] = bad.mean()
            else:
                q_fix[d, c] = float(o.mean() > tau)
    res[f"tau={fac}xO_MAX"] = {"q_thr": ev(q_thr), "q_fix": ev(q_fix)}
res["mean_o"] = ev(OJ.mean(-1))
json.dump(res, open(q.OUT / "readout_fix_real.json", "w"), indent=1); print(json.dumps(res, indent=1))
