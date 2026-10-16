"""The fixed readout (q_fix: fork -> disturbed-group share; no fork -> 1 if mean outside-hand change > tau else 0) on
IMAGINED futures from each world model, same 75 risky decisions x 13 policy plans; tau = 0.25 / 0.5 x O_MAX.
Models: deterministic closed-loop (model_US), counterfactual (model_CF), 4 ensemble members + their mean, hybrid HYB_CF."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb, step3_world_model as wm, hyb_monitor as hm
C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
O_MAX = json.loads((q.ROOT / "results/jenga/cbf_calibration.json").read_text())["O_MAX"]
R = np.array([[T[q._hash(np.asarray(p, np.float32)[1:9])] for p in r["plans"][:13]] for r in rows])
def qfix(e10, e30, z0, P, site, tau):
    ok10, _, _ = cf.split_test_gpu(e10); ok30, lab, _ = cf.split_test_gpu(e30)
    out = []
    for c in range(len(P)):
        pts = np.vstack([site[None], P[c][1:16, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        n = e30.shape[1]
        o = (((e30[c].reshape(n, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()).cpu().numpy()
        L = lab[c].cpu().numpy().astype(bool)
        row = []
        for t in tau:
            if bool(ok10[c] & ok30[c]) and not (L.all() or (~L).all()):
                bad = L if o[L].mean() > o[~L].mean() else ~L; row.append(bad.mean())
            else:
                row.append(float(o.mean() > t))
        out.append(row)
    return np.array(out)                                                   # (C, len(tau))
TAU = [0.25 * O_MAX, 0.5 * O_MAX]
mon = cf.Monitor()
models = {"det_closedloop": "results/program_b/closedloop_v1/model/model_US.pt", "det_cf": "results/program_b/cf_v1/model/model_CF.pt"}
models.update({f"ens{k}": f"results/program_b/ens_v1/model_ENS{k}.pt" for k in range(4)})
S = {}
for name, path in models.items():
    mon.model = wm.WorldModel.load(path); mon.model.net.eval()
    out = []
    for r in rows:
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda"); P = np.asarray(r["plans"], np.float32)[:13]
        bar = q.Barrier(mon, z0, 1000 + i, 64, 1.0)
        with torch.no_grad():
            e10, e30 = bar.endpoints(torch.as_tensor(P, device="cuda"))
        out.append(qfix(e10, e30, z0, P, C["site"][i], TAU)); torch.cuda.empty_cache()
    S[name] = np.stack(out); print(name, "done", flush=True)
hy = hm.HybMonitor(); out = []
for r in rows:
    i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda"); P = np.asarray(r["plans"], np.float32)[:13]
    es = []
    for c0 in range(0, 13, 4):
        idx = list(range(c0, min(13, c0 + 4)))
        acts = np.concatenate([hy.actions(P[c], 64) for c in idx])
        a = hy.det._act(torch.as_tensor(acts, device="cuda").reshape(len(idx) * 64, wm.STEPS, -1))
        g = torch.Generator(device="cuda").manual_seed(5000 + 100 * i + c0)
        es.append(hy.hy.sample(hy.head, hy.det, z0[None].expand(len(idx) * 64, -1, -1, -1).contiguous(), a, hy.rscale, g).reshape(len(idx), 64, 2, -1))
    e = torch.cat(es)
    out.append(qfix(e[:, :, 0], e[:, :, 1], z0, P, C["site"][i], TAU)); torch.cuda.empty_cache()
S["hybrid"] = np.stack(out); print("hybrid done", flush=True)
S["ensemble_mean"] = np.mean([S[f"ens{k}"] for k in range(4)], 0)
def ev(X):
    sp = [spearmanr(X[d], R[d]).correlation for d in range(len(R)) if np.ptp(X[d]) > 0 and np.ptp(R[d]) > 0]
    pick = R[np.arange(len(R)), np.argmin(X + 1e-6 * np.arange(13), 1)]
    return {"spearman": round(float(np.mean(sp)), 3), "pick_real_risk": round(float(pick.mean()), 4)}
res = {"random": round(float(R[:, 1:].mean()), 4), "oracle": round(float(R.min(1).mean()), 4), "ceiling_real_futures_tau0.5": 0.0904}
for name, X in S.items():
    res[name] = {f"tau={f}xO_MAX": ev(X[..., k]) for k, f in enumerate((0.25, 0.5))}
np.savez(q.OUT / "readout_fix_imagined.npz", **S, R=R)
json.dump(res, open(q.OUT / "readout_fix_imagined.json", "w"), indent=1); print(json.dumps(res, indent=1))
