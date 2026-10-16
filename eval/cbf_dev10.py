"""dev10 (results/jenga/cbf_dev10_protocol.json): optimisation-only filter on model_OIL (cbfmoil) vs the selection filter
on the same model (cbfmsoil), episodes 12075-12149; dev8 record / cbfm / cbfmo as secondary references; replay check."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)
A = {k: load(t) for k, t in (("record_dev8", "paired_record_dev8"), ("cbfm_dev8", "paired_cbfm_dev8"), ("cbfmo_dev8", "paired_cbfmo_dev8"),
                            ("record", "paired_record_dev10"), ("cbfmoil", "paired_cbfmoil_dev10"), ("cbfmsoil", "paired_cbfmsoil_dev10"))}
E = sorted(set.intersection(*(set(v) for v in A.values())))
R, R2 = A["record_dev8"], A["record"]
same = sum(R[e]["chunks"] == R2[e]["chunks"] and abs(R[e]["max_lift_mm"] - R2[e]["max_lift_mm"]) < 0.5 and top(R[e]) == top(R2[e]) for e in E)
out = {"episodes": len(E), "record_rerun_identical": same}
for name, f in A.items():
    if name == "record":
        continue
    iv = [x for e in E for x in f[e]["log"] if x.get("intervened")]
    out[name] = {"topples": sum(top(f[e]) for e in E), "success": sum(succ(f[e]) for e in E), "interventions": len(iv)}
    ch = [x["change_mm"] for x in iv if x.get("change_mm") is not None]
    if ch:
        out[name]["median_change_mm"] = round(float(np.median(ch)), 1)
    if iv:
        out[name]["median_s"] = round(float(np.median([x["seconds"] for x in iv])), 1)
O, S = A["cbfmoil"], A["cbfmsoil"]
out["cbfmoil_vs_cbfmsoil"] = {"opt_avoids": sum(top(S[e]) and not top(O[e]) for e in E),
                              "opt_suffers": sum(top(O[e]) and not top(S[e]) for e in E)}
t_o, t_s, s_o, s_s = out["cbfmoil"]["topples"], out["cbfmsoil"]["topples"], out["cbfmoil"]["success"], out["cbfmsoil"]["success"]
out["not_worse (adopt)"] = t_o <= t_s + 3 and s_o >= s_s - 8
out["better"] = t_o <= t_s - 4
(ROOT / "results/jenga/cbf_dev10_result.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
