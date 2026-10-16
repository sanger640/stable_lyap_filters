"""Final test 2 analysis exactly as pre-registered (results/jenga/cbf_final2_protocol.json)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parent.parent
load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)
rec, A, M = load("paired_record_final2"), load("paired_cbf_final2"), load("paired_cbfm_final2")
E = sorted(set(rec) & set(A) & set(M))
def cmp(f, g):
    b = sum(top(g[e]) and not top(f[e]) for e in E); c = sum(top(f[e]) and not top(g[e]) for e in E)
    return b, c, (binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0)
out = {"episodes": len(E), "expected": 600}
for name, f in (("record", rec), ("cbf", A), ("cbfm", M)):
    iv = [x for e in E for x in f[e]["log"] if x.get("intervened")]
    out[name] = {"topples": sum(top(f[e]) for e in E), "success": sum(succ(f[e]) for e in E), "interventions": len(iv),
                 "median_s_per_intervention": round(float(np.median([x["seconds"] for x in iv])), 1) if iv else None}
b, c, p = cmp(M, rec); out["primary_cbfm_vs_record"] = {"prevented": b, "caused": c, "p_one_sided": p}
b2, c2, p2 = cmp(M, A); out["secondary_cbfm_vs_cbf"] = {"cbfm_avoids_cbf_topple": b2, "cbfm_topples_cbf_avoids": c2, "p_one_sided": p2}
b3, c3, p3 = cmp(A, rec); out["reference_cbf_vs_record"] = {"prevented": b3, "caused": c3, "p_one_sided": p3}
out["PASS"] = bool(p < 0.05 and out["cbfm"]["success"] >= out["record"]["success"] - 15)
(ROOT / "results/jenga/cbf_final2_result.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
