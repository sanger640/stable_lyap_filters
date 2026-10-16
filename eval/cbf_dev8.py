"""dev8 analysis (results/jenga/cbf_dev8_protocol.json): record vs cbfm (final selection filter) vs cbfmo (CBF optimisation)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parent.parent
load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)
rec, S, O = load("paired_record_dev8"), load("paired_cbfm_dev8"), load("paired_cbfmo_dev8")
E = sorted(set(rec) & set(S) & set(O))
out = {"episodes": len(E), "expected": 600}
for name, f in (("record", rec), ("cbfm", S), ("cbfmo", O)):
    iv = [x for e in E for x in f[e]["log"] if x.get("intervened")]
    b = sum(top(rec[e]) and not top(f[e]) for e in E); c = sum(top(f[e]) and not top(rec[e]) for e in E)
    out[name] = {"topples": sum(top(f[e]) for e in E), "success": sum(succ(f[e]) for e in E), "interventions": len(iv),
                 "prevented_vs_record": b, "caused_vs_record": c,
                 "median_s_per_intervention": round(float(np.median([x["seconds"] for x in iv])), 1) if iv else None,
                 "median_change_mm": round(float(np.median([x["change_mm"] for x in iv if "change_mm" in x])), 1) if iv else None}
b = sum(top(S[e]) and not top(O[e]) for e in E); c = sum(top(O[e]) and not top(S[e]) for e in E)
out["cbfmo_vs_cbfm"] = {"cbfmo_avoids": b, "cbfmo_suffers": c, "p_cbfmo_better": binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else None}
nw = out["cbfmo"]["topples"] <= out["cbfm"]["topples"] + 3 and out["cbfmo"]["success"] >= out["cbfm"]["success"] - 8
out["not_worse"] = nw; out["better"] = out["cbfmo"]["topples"] <= out["cbfm"]["topples"] - 4
out["decision"] = "adopt the CBF optimisation filter; pre-register its final test" if nw else "keep the selection filter; diagnose the gap"
(ROOT / "results/jenga/cbf_dev8_result.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
