"""LSF-pub offline gate (results/jenga/lsf_pub_protocol.json): the 2026-10-21 re-audit criteria on lsf_audit / lsf_diag outputs.
usage: python3 eval/lsf_pub_gate.py value_pub_paper value_pub_code -> results/jenga/lsf_pub_gate.json"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
out = {}
for stem in sys.argv[1:]:
    a = json.loads((ROOT / f"results/jenga/lsf_audit_{stem}.json").read_text())
    d = json.loads((ROOT / f"results/jenga/lsf_value_diag_{stem}.json").read_text())
    r = {"actor_saturated_share": a["actor"]["actor_saturated_share"], "V_after_actor": a["actor"]["V_after_plan_median"],
         "V_after_hold": a["hold"]["V_after_plan_median"], "V_after_policy": a["policy"]["V_after_plan_median"],
         "AUROC_minusV_vs_any_topple": d["AUROC_minusV_vs_any_topple"], "V_quantiles": d.get("V_quantiles")}
    r["pass"] = bool(r["actor_saturated_share"] < 0.20 and r["V_after_actor"] >= r["V_after_hold"] and r["AUROC_minusV_vs_any_topple"] >= 0.70)
    out[stem] = r
(ROOT / "results/jenga/lsf_pub_gate.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
