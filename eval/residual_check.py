"""Exploratory: where the adopted 3x filter's residual failures sit in a closed-loop test (default: scl, 600 episodes).
Per neighbour-topple episode of the filter arm: did the filter act in the 8 chunks before the topple start, how early
(lead of the first intervention), was the toppling chunk itself filtered, and with what fix (certified / hold); whether the
paired policy-alone episode also toppled (prevented vs caused vs both). Per time-out episode: interventions, holds, fix
progress. usage: CL_TAG=scl ARM=cbfg3m python3 eval/residual_check.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAG, ARM = os.environ.get("CL_TAG", "scl"), os.environ.get("ARM", "cbfg3m")
A = json.loads((ROOT / f"results/jenga/outcome_audit_{TAG}.json").read_text())
F = {r["ep"]: r for r in A[f"paired_{ARM}_{TAG}"]["rows"]}
P = {r["ep"]: r for r in A[f"paired_record_{TAG}"]["rows"]}
D = ROOT / f"results/program_b/policy_rollouts/paired_{ARM}_{TAG}"


def log(e):
    return json.loads((D / f"ep{e}.json").read_text())["log"]


top, tout = [], []
for e, r in F.items():
    L = log(e); iv = {c["chunk"]: c for c in L if c.get("intervened")}
    if r["logged_neighbour_toppled"]:
        kt = next((c["chunk"] for c in L if c.get("topple_started")), None)
        pre = sorted(k for k in iv if kt is not None and kt - 8 <= k <= kt)
        top.append({"ep": e, "kt": kt, "policy_also_toppled": P[e]["logged_neighbour_toppled"], "acted_within_8": bool(pre),
                    "lead_first": (kt - pre[0]) if pre else None, "topple_chunk_filtered": kt in iv,
                    "kinds_within_8": [iv[k].get("fix_kind", iv[k].get("kind")) for k in pre],
                    "interventions_total": len(iv), "h_nom_at_topple": next((c["h_nom"] for c in L if c["chunk"] == kt), None)})
    elif not r["extracted"] and not r["red_fell_before_extraction"]:
        tout.append({"ep": e, "interventions": len(iv), "holds": sum(c.get("fix_kind") == "hold" for c in iv.values()),
                     "progress_median": float(np.median([c["progress"] for c in iv.values() if "progress" in c] or [np.nan])),
                     "policy_alone_succeeded": P[e]["v4_success"], "max_lift_mm": r["max_lift_mm"]})

T = top
out = {"tag": TAG, "arm": ARM, "topples": len(T),
       "caused_by_filter_arm_only": sum(not t["policy_also_toppled"] for t in T),
       "acted_within_8_before": sum(t["acted_within_8"] for t in T),
       "lead_hist": {str(k): sum(t["lead_first"] == k for t in T) for k in range(9)},
       "topple_chunk_was_filtered": sum(t["topple_chunk_filtered"] for t in T),
       "never_acted_before": sum(not t["acted_within_8"] for t in T),
       "h_nom_at_topple_chunk": [t["h_nom_at_topple"] for t in T],
       "timeouts": len(tout), "timeout_policy_alone_succeeded": sum(t["policy_alone_succeeded"] for t in tout),
       "timeout_interventions_median": float(np.median([t["interventions"] for t in tout])) if tout else None,
       "timeout_holds_median": float(np.median([t["holds"] for t in tout])) if tout else None,
       "timeout_fix_progress_median": float(np.nanmedian([t["progress_median"] for t in tout])) if tout else None,
       "rows_topple": T, "rows_timeout": tout}
(ROOT / f"results/jenga/deep_dive/residual_{TAG}_{ARM}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.startswith("rows")}, indent=1))
