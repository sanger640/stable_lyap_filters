"""Trigger quality of the fixed readout over the 361 cached dev3A decisions (real risk of the policy chunk cached):
compare the barrier trigger (h only), the final-test trigger, and q_fix-based triggers (policy plan vs holding still)."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CBF_WM", os.path.abspath("results/program_b/closedloop_v1/model/model_US.pt"))
import cbf_qp as q, cem_filter as cf, cbf_filter as cbff, inspector_data as insp, torch
from sklearn.metrics import roc_auc_score
C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
mon = cf.Monitor(seed=321); cbf = cbff.CBF(mon); sim = insp._sim()
O = cbff.CFG["O_MAX"]; rows = []
for i, k in enumerate(K):
    sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
    z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
    hold = np.repeat(np.r_[C["target"][i], 0.0][None], 16, 0).astype(np.float32)
    qv = cbf.qfix(sim, z0, np.stack([C["plan16"][i].astype(np.float32), hold]))
    rows.append({"risk": T[q._hash(C["nominal"][i])], "q_pol": float(qv[0]), "q_hold": float(qv[1]), "h": k["h_nom"], "h_now": k["h_now"], "o": k["o_nom"]})
risk = np.array([r["risk"] for r in rows]); y = risk >= 2 / 32
def rate(flag):
    flag = np.asarray(flag, bool)
    return {"flagged": int(flag.sum()), "caught_risky": round(float(flag[y].mean()), 3), "false_on_safe": round(float(flag[~y].mean()), 3),
            "risk_flagged": round(float(risk[flag].mean()), 3) if flag.any() else None, "risk_unflagged": round(float(risk[~flag].mean()), 3)}
qp = np.array([r["q_pol"] for r in rows]); qh = np.array([r["q_hold"] for r in rows])
h = np.array([r["h"] for r in rows]); hn = np.array([r["h_now"] for r in rows]); o = np.array([r["o"] for r in rows])
res = {"decisions": len(rows), "risky": int(y.sum()),
       "barrier_only": rate(h < np.minimum(hn, 1)), "final_test_trigger": rate((h < np.minimum(hn, 1)) | (o > O)),
       "qfix_policy_ge_0.5": rate(qp >= 0.5), "qfix_policy_gt_hold": rate(qp > qh + 0.05), "qfix_policy_gt_0": rate(qp > 0),
       "barrier_or_qfix_gt_hold": rate((h < np.minimum(hn, 1)) | (qp > qh + 0.05)),
       "auc_qfix_policy": round(float(roc_auc_score(y, qp)), 3), "auc_minus_h": round(float(roc_auc_score(y, -h)), 3)}
json.dump(res, open(q.OUT / "qfix_trigger.json", "w"), indent=1); print(json.dumps(res, indent=1))
