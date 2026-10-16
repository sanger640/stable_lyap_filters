"""Calibration of the world model's imagined outcomes vs reality, on the 75 risky dev decisions x 21 plans (policy plan,
12 policy samples, 8 nudges; real risk cached, 32 futures). Imagined topple fraction q = share of 64 imagined futures in
the split group with the larger change away from the hand (0 if no split); also the soft threshold version.
Reliability: bins of imagined q -> mean real topple rate; plus ECE and spearman across all plans."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb
C, K = q.load_cache()
T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
O_MAX = json.loads((q.ROOT / "results/jenga/cbf_calibration.json").read_text())["O_MAX"]
os.environ["CBF_WM"] = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "results/program_b/cf_v1/model/model_CF.pt")
mon = cf.Monitor()
Q, QS, R = [], [], []
for r in rows:
    i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
    P = np.asarray(r["plans"], np.float32)
    bar = q.Barrier(mon, z0, 1000 + i, 64, 1.0)
    with torch.no_grad():
        _, e30 = bar.endpoints(torch.as_tensor(P, device="cuda")); _, lab, _ = cf.split_test_gpu(e30)
        ok, _, _ = cf.split_test_gpu(e30)
    for c in range(len(P)):
        pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        o_j = (((e30[c].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum())
        L = lab[c].bool()
        if bool(ok[c]) and not (L.all() or (~L).all()):
            bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
            Q.append(float(bad.float().mean()))
        else:
            Q.append(0.0)
        QS.append(float((o_j > O_MAX).float().mean()))
        R.append(T[q._hash(P[c][1:9])])
Q, QS, R = map(np.array, (Q, QS, R))
def rel(x):
    edges = [0, 1e-9, 0.1, 0.2, 0.35, 0.5, 1.01]; out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.any(): out.append({"imagined": f"[{a:.2f},{b:.2f})", "n": int(m.sum()), "mean_imagined": round(float(x[m].mean()), 3), "real_rate": round(float(R[m].mean()), 3)})
    ece = sum(o["n"] * abs(o["mean_imagined"] - o["real_rate"]) for o in out) / len(x)
    return {"bins": out, "ECE": round(float(ece), 3), "spearman_all": round(float(spearmanr(x, R).correlation), 3)}
res = {"plans": len(R), "mean_real": round(float(R.mean()), 3), "q_fork_group": rel(Q), "q_threshold_OMAX": rel(QS), "model": os.environ["CBF_WM"]}
json.dump(res, open(q.OUT / "reliability.json", "w"), indent=1); print(json.dumps(res, indent=1))
