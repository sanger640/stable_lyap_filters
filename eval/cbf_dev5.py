"""dev5 analysis (protocol results/jenga/cbf_dev5_protocol.json): record vs cbf (final-test filter) vs cbfq (fixed-readout selection; single
CBF optimisation), episodes 5000-5149, all arms of an episode in one cluster job. Reads results/jenga/paired_*_dev5.jsonl
(merged from the clusters' per-episode files)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parent.parent
load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)
rec, A, O = load("paired_record_dev5"), load("paired_cbf_dev5"), load("paired_cbfq_dev5")
E = sorted(set(rec) & set(A) & set(O))
out = {"episodes": len(E)}
for name, f in (("cbf", A), ("cbfq", O)):
    b = sum(top(rec[e]) and not top(f[e]) for e in E); c = sum(top(f[e]) and not top(rec[e]) for e in E)
    L = [x for e in E for x in f[e]["log"]]; iv = [x for x in L if x.get("intervened")]; trig = [x for x in L if x.get("kind") in ("qfix",) or x.get("intervened")]
    out[name] = {"topples": sum(top(f[e]) for e in E), "prevented": b, "caused": c,
                 "p_vs_record": round(binomtest(b, b + c, 0.5, alternative="greater").pvalue, 4) if b + c else None,
                 "success": sum(succ(f[e]) for e in E), "interventions": len(iv),
                 "median_s_per_intervention": round(float(np.median([x["seconds"] for x in iv])), 1) if iv else None}
out["record"] = {"topples": sum(top(rec[e]) for e in E), "success": sum(succ(rec[e]) for e in E)}
bo = sum(top(A[e]) and not top(O[e]) for e in E); co = sum(top(O[e]) and not top(A[e]) for e in E)
out["cbfq_vs_cbf"] = {"cbfq_avoids_cbf_topple": bo, "cbfq_topples_cbf_avoids": co,
                        "p": round(binomtest(bo, bo + co, 0.5, alternative="greater").pvalue, 4) if bo + co else None}
d = out["cbf"]["topples"] - out["cbfq"]["topples"]
ok_succ = out["cbfq"]["success"] >= out["cbf"]["success"] - 8
out["decision"] = ("take cbfq to a fresh final test" if d >= 5 and ok_succ else
                   "not distinguishable on 300 episodes: decide with the user" if d > 0 and ok_succ else "keep the final-test filter")
(ROOT / "results/jenga/cbf_dev5_result.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
