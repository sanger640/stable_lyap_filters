"""LSF dev eps sweep (docs/LATENT_SAFETY_FILTER.md, spec + amendment fixed before any dev episode ran): per arm, replay-
audited neighbour topples / v4 / v4b on the paired episodes 15000-15099; selection: the eps with the fewest topples among
those whose v4 is not more than 5 pp below the policy alone; ties -> larger eps.
usage: python3 eval/lsf_dev_analysis.py  -> results/jenga/lsf_dev_result.json"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "results/jenga/outcome_audit_lsfdev.json").read_text())
ARMS = ["record"] + [f"lsfe{e:03d}" for e in (0, 5, 10, 15, 20, 30)]
R = {a: {r["ep"]: r for r in A[f"paired_{a}_lsfdev"]["rows"]} for a in ARMS}
EPS = sorted(set.intersection(*(set(v) for v in R.values())))
cnt = {a: {"topples": sum(R[a][e]["logged_neighbour_toppled"] for e in EPS), "v4": sum(R[a][e]["v4_success"] for e in EPS),
           "v4b": sum(R[a][e]["v4b_success"] for e in EPS)} for a in ARMS}
pol_v4 = cnt["record"]["v4"] / len(EPS)
ok = [a for a in ARMS[1:] if cnt[a]["v4"] / len(EPS) >= pol_v4 - 0.05]
sel = min(ok, key=lambda a: (cnt[a]["topples"], -int(a[4:]))) if ok else None
out = {"episodes": len(EPS), "counts": cnt, "eligible_v4_within_5pp": ok, "selected": sel,
       "selected_eps": int(sel[4:]) / 100 if sel else None,
       "replay_flag_matches": {a: A[f"paired_{a}_lsfdev"]["replay_topple_flag_matches"] for a in ARMS}}
(ROOT / "results/jenga/lsf_dev_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
