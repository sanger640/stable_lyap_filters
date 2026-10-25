"""pconf analysis as pre-registered (results/jenga/pconf_protocol.json), written before any pconf episode ran.
C1 (confirmation): cbfps3m vs record: fewer neighbour topples (exact McNemar one-sided p < 0.05) AND v4 higher (paired
bootstrap one-sided 95% lower bound > 0). C2 (margin): cbfps4m vs cbfps3m: fewer topples (same test) AND v4 non-inferior
(lower bound >= -3 pp). Same helpers as eval/steer_analysis.py, bootstrap seed 20261025.
usage: python3 eval/pconf_analysis.py  (after outcome_audit.py runs paired_{record,cbfps3m,cbfps4m}_pconf)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ["CL_TAG"] = "pconf"; os.environ["CL_ARMS"] = "record cbfps3m cbfps4m"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
src = (ROOT / "eval/steer_analysis.py").read_text()
ns = {"__file__": str(ROOT / "eval/steer_analysis.py")}
exec(src[:src.index("KEYS = ")], ns)                        # helpers only (vec, mcnemar_less, boot, ivstats), not its primary
ns["rng"] = np.random.default_rng(20261025)
vec, mcn, boot, ivstats, EPS = ns["vec"], ns["mcnemar_less"], ns["boot"], ns["ivstats"], ns["EPS"]
ARMS = ("record", "cbfps3m", "cbfps4m")


def first_grasp(a):
    P = ROOT / "results/program_b/policy_rollouts" / f"paired_{a}_pconf"; fg = {}
    for e in EPS:
        g = np.load(P / f"ep{e}.npz")["gripper"]; c = np.flatnonzero(g < 55); fg[e] = int(c[0]) if len(c) else None
    return fg


def anatomy(a):
    fg = first_grasp(a); to = [e for e, t in zip(EPS, vec(a, "timeout")) if t]
    succ = [fg[e] for e, s in zip(EPS, vec(a, "v4b_success")) if s and fg[e] is not None]
    return {"never_grasp": sum(fg[e] is None for e in to), "grasp_ge45": sum(fg[e] is not None and fg[e] >= 45 for e in to),
            "grasp_then_no_extract": sum(fg[e] is not None and fg[e] < 45 for e in to),
            "first_grasp_chunk_median_v4b": float(np.median(succ)) if succ else None}


KEYS = ("topple", "timeout", "failure", "red_fell_before_extraction", "red_fell_after_extraction", "v4_success", "v4b_success")
out = {"episodes": len(EPS), "counts": {a: {k: int(vec(a, k).sum()) for k in KEYS} for a in ARMS}}
c1t = mcn("cbfps3m", "record", "topple"); c1v = boot("cbfps3m", "record", "v4_success")
c2t = mcn("cbfps4m", "cbfps3m", "topple"); c2v = boot("cbfps4m", "cbfps3m", "v4_success")
out["C1_confirmation"] = {"fewer_topples": c1t, "v4_minus_policy": c1v,
                          "pass": bool(c1t["p_one_sided"] < 0.05 and c1v["lower95_one_sided_pp"] > 0)}
out["C2_margin"] = {"fewer_topples": c2t, "v4_minus_3x": c2v,
                    "pass": bool(c2t["p_one_sided"] < 0.05 and c2v["lower95_one_sided_pp"] >= -3.0)}
out["secondary"] = {
    "v4b_3x_minus_policy": boot("cbfps3m", "record", "v4b_success"), "v4b_4x_minus_3x": boot("cbfps4m", "cbfps3m", "v4b_success"),
    "v4_4x_minus_policy": boot("cbfps4m", "record", "v4_success"), "4x_fewer_topples_than_policy": mcn("cbfps4m", "record", "topple"),
    "timeouts_4x_vs_3x_more": mcn("cbfps3m", "cbfps4m", "timeout"),
    "drop_rate_per_extraction": {a: round(float((vec(a, "v4_success") * vec(a, "red_fell_after_extraction")).sum() / max(1, vec(a, "v4_success").sum())), 3) for a in ARMS},
    **{f"{a}_anatomy": anatomy(a) for a in ARMS},
    **{f"{a}_interventions": ivstats(a) for a in ARMS if a != "record"}}
(ROOT / "results/jenga/pconf_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
