import json, os, sys, numpy as np
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
sys.path[:0] = ["eval", "src", "libero"]
import outcome_audit as oa, plausibility as pl, inspector_data as insp
A = json.load(open("results/jenga/outcome_audit_m3.json"))
sim = insp._sim(); out = {}
for arm in ("cbfg3m", "record"):
    rows = A[f"paired_{arm}_m3"]["rows"]; RUN = Path(f"results/program_b/policy_rollouts/paired_{arm}_m3")
    res = []
    for r in rows:
        if r["logged_neighbour_toppled"] or not (r["red_fell_before_extraction"] or r["red_fell_after_extraction"]):
            continue
        ep = r["ep"]
        with np.load(RUN / f"ep{ep}.npz") as z:
            S = {k: z[k] for k in z.files}
        L = json.loads((RUN / f"ep{ep}.json").read_text())["log"]
        _, TT, _, _ = oa.replay(sim, S, pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1))
        df = int(np.argmax(TT >= 45)) // 8
        iv = [bool(x) for x in S["intervened"]]; al = [bool(x) for x in S["alarm"]] if "alarm" in S else None
        g = [float(x) < 50 for x in S["gripper"]]
        res.append({"ep": ep, "kind": "before" if r["red_fell_before_extraction"] else "after", "df": df, "n": len(iv),
                    "iv_window": iv[max(0, df - 3):df + 1], "closed_at_df": g[min(df, len(g) - 1)],
                    "opens_in_df_chunk": bool(g[min(df, len(g) - 1)] and df + 1 < len(g) and not g[df + 1])})
        print(arm, ep, res[-1]["kind"], df, res[-1]["iv_window"], flush=True)
    out[arm] = res
json.dump(out, open("results/jenga/deep_dive/fall_timing_m3.json", "w"))
for arm, res in out.items():
    for kind in ("before", "after"):
        rs = [x for x in res if x["kind"] == kind]
        if not rs: continue
        print(arm, kind, len(rs), "intervened at fall chunk %.2f" % np.mean([x["iv_window"][-1] for x in rs if x["iv_window"]]),
              "any in df-3..df %.2f" % np.mean([any(x["iv_window"]) for x in rs]), "closed at df %.2f" % np.mean([x["closed_at_df"] for x in rs]),
              "gripper opens next %.2f" % np.mean([x["opens_in_df_chunk"] for x in rs]))
