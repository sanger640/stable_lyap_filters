"""Irreversibility-barrier CBF dev summary (results/jenga/irr_protocol.json): arms irg / irs (tag _irrdev) vs the policy
alone (paired_record_lflsfdev), episodes 15000-15099, replay-audited outcomes; the LSF dev rule (fewest audited neighbour
topples among arms with v4 >= policy - 5 pp; ties -> fewer interventions per episode).
usage: python3 eval/irr_dev.py -> results/jenga/irr_dev.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("irg", "irs", "lfci000")


def main():
    A = json.loads((ROOT / "results/jenga/outcome_audit_lflsfdev.json").read_text())
    A.update(json.loads((ROOT / "results/jenga/outcome_audit_irrdev.json").read_text()))
    keys = {"record": "paired_record_lflsfdev", **{a: f"paired_{a}_irrdev" for a in ARMS}}
    R = {a: {r["ep"]: r for r in A[k]["rows"]} for a, k in keys.items() if k in A}
    EPS = sorted(set.intersection(*(set(v) for v in R.values())))
    q = ("logged_neighbour_toppled", "v4_success", "v4b_success", "red_fell_before_extraction", "red_fell_after_extraction")
    cnt = {}
    for a in R:
        c = {k: sum(bool(R[a][e][k]) for e in EPS) for k in q}
        c["timeout"] = sum(not R[a][e]["extracted"] and not R[a][e]["logged_neighbour_toppled"] and not R[a][e]["red_fell_before_extraction"] for e in EPS)
        if a != "record":
            L = [json.loads((ROOT / f"results/program_b/policy_rollouts/{keys[a]}/ep{e}.json").read_text())["log"] for e in EPS]
            fx = [d for l in L for d in l if d.get("intervened")]
            c.update({"interventions_per_ep": round(len(fx) / len(EPS), 2),
                      "fix_kinds": {k: sum(d.get("fix_kind") == k for d in fx) for k in ("certified", "fallback_hold", "fallback_best")},
                      "median_change_mm": float(np.median([d.get("change_mm", 0) for d in fx])) if fx else None,
                      "median_fix_s": float(np.median([d["seconds"] for d in fx])) if fx else None,
                      "trigger_share": round(len(fx) / max(1, sum(len(l) for l in L)), 4)})
        cnt[a] = c
    pv4 = cnt["record"]["v4_success"] / len(EPS)
    ok = [a for a in ARMS if a in cnt and cnt[a]["v4_success"] / len(EPS) >= pv4 - 0.05]
    sel = min(ok, key=lambda a: (cnt[a]["logged_neighbour_toppled"], cnt[a]["interventions_per_ep"])) if ok else None
    out = {"episodes": len(EPS), "counts": cnt, "eligible": ok, "selected": sel}
    (ROOT / "results/jenga/irr_dev.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
