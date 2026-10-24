"""Exploratory (LSF audit 2): is the latent safety filter's learned value a good ALARM, separate from how it is used?
On the LSF arms' own episodes (the runtime logs V_min of the policy's segment at every decision; at eps <= 0.1 the arm
almost never intervenes, so these are nearly policy-alone episodes), the early-warning metric of the monitor tests:
at a matched safe alarm rate, the share of topple episodes alarmed within the 8 chunks before the topple, >= 3 ahead.
usage: python3 eval/lsf_trigger_check.py   -> results/jenga/deep_dive/lsf_trigger_check.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/program_b/policy_rollouts"
SETS = {"h2h2": ("paired_lsfe010_h2h2", "outcome_audit_h2h2.json"), "unidev_e000": ("paired_lsfe000_unidev", "outcome_audit_unidev.json"),
        "unidev_e010": ("paired_lsfe010_unidev", "outcome_audit_unidev.json")}
RATES = (0.062, 0.113, 0.15, 0.20)


def load(run, audit):
    A = {r["ep"]: r["logged_neighbour_toppled"] for r in json.loads((ROOT / "results/jenga" / audit).read_text())[run]["rows"]}
    E = {}
    for e in A:
        f = RUN / run / f"ep{e}.json"
        if f.exists():
            L = json.loads(f.read_text())["log"]
            kt = next((c["chunk"] for c in L if c.get("topple_started")), None)
            E[e] = {"top": A[e], "kt": kt, "v": {c["chunk"]: c["lsf_vmin"] for c in L if "lsf_vmin" in c},
                    "iv": sum(bool(c.get("intervened")) for c in L)}
    return E


def evaluate(E):
    safe_v = np.array([v for x in E.values() if not x["top"] for v in x["v"].values()])
    out = {"episodes": len(E), "topple_episodes": sum(x["top"] and x["kt"] is not None for x in E.values()),
           "safe_decisions": len(safe_v), "interventions_per_episode": round(np.mean([x["iv"] for x in E.values()]), 2)}
    for r in RATES:
        thr = float(np.quantile(safe_v, r))                       # alarm iff V_min < thr (low value = unsafe)
        hit = []
        for x in E.values():
            if x["top"] and x["kt"] is not None:
                ks = [k for k in sorted(x["v"]) if x["kt"] - 8 <= k <= x["kt"] and x["v"][k] < thr]
                hit.append(bool(ks) and x["kt"] - ks[0] >= 3)
        out[f"rate_{r}"] = {"threshold": round(thr, 4), "lead_ge3_share": round(float(np.mean(hit)), 3)}
    # what the deployed eps saw: V_min quantiles on safe decisions and at the topple chunk's previous 3 decisions
    pre = [x["v"][k] for x in E.values() if x["top"] and x["kt"] is not None for k in range(x["kt"] - 3, x["kt"]) if k in x["v"]]
    out["V_safe_q05_50_95"] = [round(float(np.quantile(safe_v, q)), 3) for q in (0.05, 0.5, 0.95)]
    out["V_pretopple_q05_50_95"] = [round(float(np.quantile(pre, q)), 3) for q in (0.05, 0.5, 0.95)] if pre else None
    out["share_pretopple_below_0.1"] = round(float(np.mean(np.array(pre) < 0.1)), 3) if pre else None
    return out


if __name__ == "__main__":
    res = {k: evaluate(load(*v)) for k, v in SETS.items()}
    (ROOT / "results/jenga/deep_dive/lsf_trigger_check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
