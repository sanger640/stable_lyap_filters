"""IRRM amendment 1 offline gates (results/jenga/irrm_protocol.json).
  B: python3 eval/irrm_gate.py B   value_lf_alea_n1.pt: LF2's criteria from lsf_audit / lsf_diag (run with the matched noise)
  A: python3 eval/irrm_gate.py A   value_irr8_alea.pt: LF2's criteria AND the value of the state V(z, g) on CF3's fresh
                                   decisions (irrm1 rows): AUC(-V, pre-topple 1-3 chunks vs safe) >= 0.80, V < 0 on safe
                                   gripper-closed <= 5%, safe gripper-open <= 10%
  M: python3 eval/irrm_gate.py M   the margin net's gate from results/jenga/irrm2_result_nl8.json
-> results/jenga/irrm_{A,B,M}_gate.json (lflsf_LF2.json is not touched)"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
NAME = {"A": "value_irr8_alea.pt", "B": "value_lf_alea_n1.pt"}


def lf2(stem):
    a = json.loads((ROOT / f"results/jenga/lsf_audit_{stem}.json").read_text())
    d = json.loads((ROOT / f"results/jenga/lsf_value_diag_{stem}.json").read_text())
    out = {"actor_saturated_share": a["actor"]["actor_saturated_share"], "V_after_actor": a["actor"]["V_after_plan_median"],
           "V_after_policy": a["policy"]["V_after_plan_median"], "V_after_hold": a["hold"]["V_after_plan_median"],
           "AUROC_minusV_vs_any_topple": d["AUROC_minusV_vs_any_topple"], "V_quantiles": d["V_quantiles"], "share_V_below_0.3": d["share_V_below_0.3"]}
    out["LF2_pass"] = bool(out["actor_saturated_share"] < 0.05 and out["V_after_actor"] >= out["V_after_hold"] and out["AUROC_minusV_vs_any_topple"] >= 0.70)
    return out


def decisions(name):
    """V(z, g) on CF3's fresh decisions (the stage 1 rows) -> grading as stage 1"""
    import torch
    from sklearn.metrics import roc_auc_score
    import cf_conf as CC
    import lsf as L
    fs = sorted((ROOT / "results/jenga").glob("irrm1_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("run", "ep", "i", "kt", "k", "closed")}
    vf = L.ValueFn(name); V = np.zeros(len(R["ep"])); cache = {}
    for n in range(len(V)):
        run_ = CC.RUNS[int(R["run"][n])]; stem = f"ep{int(R['ep'][n])}"; i = int(R["i"][n])
        if (run_, stem) not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz = d["z"].astype(np.float32)
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cache[(run_, stem)] = (hz, plan16, target)
        hz, plan16, target = cache[(run_, stem)]
        with torch.no_grad():
            z = torch.as_tensor(hz[i][None], device="cuda").float(); g = torch.as_tensor(np.r_[target[i], plan16[i][0, 3]][None], dtype=torch.float32, device="cuda")
            V[n] = float(vf.V(z, g))
    kt, k, cl = R["kt"], R["k"], R["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0; ok = pre | safe
    out = {"decisions": int(len(V)), "auc_minusV": round(float(roc_auc_score(pre[ok], -V[ok])), 4),
           "V_below0": {"pre": round(float((V[pre] < 0).mean()), 4), "safe_closed": round(float((V[safe & cl] < 0).mean()), 4),
                        "safe_open": round(float((V[safe & ~cl] < 0).mean()), 4)}}
    out["decisions_pass"] = bool(out["auc_minusV"] >= 0.80 and out["V_below0"]["safe_closed"] <= 0.05 and out["V_below0"]["safe_open"] <= 0.10)
    return out


def main(which):
    if which == "M":
        r = json.loads((ROOT / "results/jenga/irrm2_result_nl8.json").read_text())
        out = {"held_spearman": r["held_spearman"], "held_auc_hat_vs_true_flag": r["held_auc_hat_vs_true_flag"]}
        out["M_pass"] = bool(r["held_spearman"] >= 0.6 and (r["held_auc_hat_vs_true_flag"] or 0) >= 0.85)
    else:
        out = lf2(NAME[which][:-3])
        if which == "A":
            out["cf3_decisions"] = decisions(NAME[which])
            out["A_pass"] = bool(out["LF2_pass"] and out["cf3_decisions"]["decisions_pass"])
        else:
            out["cf3_decisions_reported"] = decisions(NAME[which])                  # reported for comparison, not gated
            out["B_pass"] = out["LF2_pass"]
    (ROOT / f"results/jenga/irrm_{which}_gate.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
