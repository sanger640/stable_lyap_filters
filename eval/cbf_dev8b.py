"""dev8b: cbfmo3 (alpha 0.3) vs dev8's record / cbfm / cbfmo on episodes 12000-12074; also checks the record rerun matches."""
import json
from pathlib import Path
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parent.parent
load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)
R, S, O5, R2, O3 = (load(t) for t in ("paired_record_dev8", "paired_cbfm_dev8", "paired_cbfmo_dev8", "paired_record_dev8c", "paired_cbfmo3_dev8c"))
E = sorted(set(R) & set(S) & set(O5) & set(R2) & set(O3))
same = sum(R[e]["chunks"] == R2[e]["chunks"] and abs(R[e]["max_lift_mm"] - R2[e]["max_lift_mm"]) < 0.5 and top(R[e]) == top(R2[e]) for e in E)
out = {"episodes": len(E), "record_rerun_identical": same}
for name, f in (("record", R), ("cbfm", S), ("cbfmo_a05", O5), ("cbfmo_a03", O3)):
    iv = [x for e in E for x in f[e]["log"] if x.get("intervened")]
    out[name] = {"topples": sum(top(f[e]) for e in E), "success": sum(succ(f[e]) for e in E), "interventions": len(iv)}
    if iv and "change_mm" in iv[0]:
        import numpy as np
        out[name]["median_change_mm"] = round(float(np.median([x["change_mm"] for x in iv if "change_mm" in x])), 1)
        out[name]["median_s"] = round(float(np.median([x["seconds"] for x in iv])), 1)
b = sum(top(S[e]) and not top(O3[e]) for e in E); c = sum(top(O3[e]) and not top(S[e]) for e in E)
out["a03_vs_cbfm"] = {"a03_avoids": b, "a03_suffers": c}
out["meets_small_rule"] = out["cbfmo_a03"]["topples"] <= out["cbfm"]["topples"] + 1 and out["cbfmo_a03"]["success"] >= out["cbfm"]["success"] - 2
(ROOT / "results/jenga/cbf_dev8b_result.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
