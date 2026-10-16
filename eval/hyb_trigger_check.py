"""Trigger quality over all 361 cached dev3A decisions (211 where the final-test filter acted + 150 calm): does the
hybrid's verdict on the policy's plan pick out the really risky moments better than the final-test trigger?
Real risk = policy chunk, 32 exact-sim futures (cached). Risky = >= 2/32 toppling futures."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from sklearn.metrics import roc_auc_score
import torch, hyb_monitor as hm
C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
mon = hm.HybMonitor()
rows = []
for i, k in enumerate(K):
    z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
    hold = np.repeat(np.r_[C["target"][i], 0.0][None], 16, 0).astype(np.float32)
    fork, qq, ob = mon.score(z0, np.stack([C["plan16"][i].astype(np.float32), hold]), C["site"][i], 7000 + i)
    rows.append({"i": i, "acted": k["acted"], "risk": T[q._hash(C["nominal"][i])], "fork": bool(fork[0]), "q": float(qq[0]),
                 "q_hold": float(qq[1]), "fork_hold": bool(fork[1]), "obar": float(ob[0]), "h_nom": k["h_nom"]})
    if i % 50 == 0: print("decision", i, flush=True)
risk = np.array([r["risk"] for r in rows]); y = risk >= 2 / 32
def rate(flag):
    flag = np.asarray(flag, bool)
    return {"flagged": int(flag.sum()), "caught_risky": round(float(flag[y].mean()), 3), "false_on_safe": round(float(flag[~y].mean()), 3),
            "mean_risk_flagged": round(float(risk[flag].mean()), 3) if flag.any() else None, "mean_risk_unflagged": round(float(risk[~flag].mean()), 3)}
res = {"decisions": len(rows), "risky": int(y.sum()),
       "final_test_trigger": rate([r["acted"] for r in rows]),
       "hybrid_fork": rate([r["fork"] for r in rows]),
       "hybrid_q_above_hold": rate([r["q"] > r["q_hold"] + 0.05 for r in rows]),
       "auc": {"hybrid_q": round(roc_auc_score(y, [r["q"] for r in rows]), 3), "hybrid_obar": round(roc_auc_score(y, [r["obar"] for r in rows]), 3),
               "det_barrier_minus_h": round(roc_auc_score(y, [-r["h_nom"] for r in rows]), 3)}}
json.dump({"summary": res, "rows": rows}, open(q.OUT / "hyb_trigger.json", "w"), indent=1); print(json.dumps(res, indent=1))
