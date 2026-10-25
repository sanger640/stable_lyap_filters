"""Exploratory (post-hoc, 2026-10-25): why does psel (cbfps3m) drop the red block after extraction more often than
cbfg3m (89 vs 59 of 600)? For every extracted episode of both arms, replays it (outcome_audit.replay) and records the
extraction / fall chunk, the interventions between them, the gripper change the fix made vs the policy's plan, and the
red block's lift when it fell.

usage: PD_SHARD=i/K python3 eval/psel_drop_check.py (rows -> deep_dive/psel_drop_rows_i.jsonl); then
       python3 eval/psel_drop_check.py merge -> results/jenga/deep_dive/psel_drop_check.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import outcome_audit as oa  # noqa: E402  (sets MUJOCO_GL / HARD_MASS)


DD = ROOT / "results/jenga/deep_dive"


def rows_main(tag="psel", arms=("cbfg3m", "cbfps3m")):
    import inspector_data as insp
    import plausibility as pl
    i, K = (int(v) for v in os.environ.get("PD_SHARD", "0/1").split("/"))
    sim = insp._sim(); aud = json.loads((ROOT / f"results/jenga/outcome_audit_{tag}.json").read_text())
    fo = open(DD / f"psel_drop_rows_{i}.jsonl", "a")
    for arm in arms:
        run = f"paired_{arm}_{tag}"
        logs = {(r := json.loads(l))["episode"]: r["log"] for l in (ROOT / f"results/jenga/{run}.jsonl").read_text().splitlines()}
        for x in aud[run]["rows"]:
            if not x["v4_success"] or x["ep"] % K != i:
                continue
            ep = x["ep"]
            with np.load(oa.RUNS / run / f"ep{ep:03d}.npz") as f:
                S = {k: f[k] for k in f.files}
            L, TT, NT, XY = oa.replay(sim, S, pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1))
            ext = (L >= 0.02) | (XY[:, 0] <= -0.02); i_ext = int(np.argmax(ext))
            i_fall = int(np.argmax(TT >= 45)) if (TT >= 45).any() else None
            ce = i_ext // 8; cf = i_fall // 8 if i_fall is not None else len(S["chunk_index"]) - 1
            log = logs[ep]; post = [g for g in log if ce <= g["chunk"] <= cf and g.get("intervened")]
            dgrip = [float(np.abs(S["executed"][g["chunk"]][:, 3] - S["nominal"][g["chunk"]][:, 3]).max()) for g in post]
            row = ({"ep": ep, "fell": i_fall is not None, "ext_chunk": ce, "fall_chunk": i_fall // 8 if i_fall is not None else None,
                         "lift_at_fall_mm": round(float(L[i_fall] * 1000), 1) if i_fall is not None else None,
                         "post_interventions": len(post), "post_kinds": [g.get("fix_kind") for g in post],
                         "post_max_grip_change": round(max(dgrip), 3) if dgrip else 0.0,
                         "interv_in_last2_before_fall": sum(1 for g in post if i_fall is not None and cf - 2 <= g["chunk"] <= cf)})
            fo.write(json.dumps({"arm": arm, **row}) + "\n"); fo.flush()


def merge(arms=("cbfg3m", "cbfps3m")):
    allr = [json.loads(l) for f in sorted(DD.glob("psel_drop_rows_*.jsonl")) for l in f.read_text().splitlines()]
    out = {}
    for arm in arms:
        rows = sorted((r for r in allr if r["arm"] == arm), key=lambda r: r["ep"])
        fell = [r for r in rows if r["fell"]]; kept = [r for r in rows if not r["fell"]]
        m = lambda R, k: round(float(np.mean([r[k] for r in R])), 2) if R else None
        out[arm] = {"extracted": len(rows), "fell_after": len(fell),
                    "post_interventions_mean": {"fell": m(fell, "post_interventions"), "kept": m(kept, "post_interventions")},
                    "frac_with_post_intervention": {"fell": m([{"v": r["post_interventions"] > 0} for r in fell], "v"),
                                                    "kept": m([{"v": r["post_interventions"] > 0} for r in kept], "v")},
                    "frac_intervened_within_2_chunks_before_fall": m([{"v": r["interv_in_last2_before_fall"] > 0} for r in fell], "v"),
                    "frac_post_grip_change_gt_0.1": {"fell": m([{"v": r["post_max_grip_change"] > 0.1} for r in fell], "v"),
                                                    "kept": m([{"v": r["post_max_grip_change"] > 0.1} for r in kept], "v")},
                    "lift_at_fall_mm_median": float(np.median([r["lift_at_fall_mm"] for r in fell])) if fell else None,
                    "rows": rows}
    summ = {a: {k: v for k, v in out[a].items() if k != "rows"} for a in out}
    (DD / "psel_drop_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    merge() if sys.argv[1:] == ["merge"] else rows_main()
