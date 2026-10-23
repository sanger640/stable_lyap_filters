"""Gate for the full-token DINO-WM (docs/FILTER_DEEP_DIVE.md step 2): does it RANK candidate chunks from the same state by
real risk better than model_OIL? Same chunks, same plans (chunk + hold tail) and same pair rule as eval/rank_check.py;
scored by tokmon.TokMon (action-noise draws only), 64 futures, readouts h (fork scale), g (sensitivity gain), o (disturbance).
usage: python3 eval/tok_rank.py <set> <model.pt> <tag> [analyse]
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
D = ROOT / "results/jenga/deep_dive"


def main(name, model_path, tag):
    import cem_filter as cf
    import inspector_data as insp
    import rank_check as rc
    import tokfilter as tf
    import tokmon
    C, cand = rc.chunks_for(name); mon = tokmon.TokMon(model_path); fr = cf.Monitor.__new__(cf.Monitor); fr.renderer = None
    sim = insp._sim(); rows = []; lim = int(os.environ.get("TOK_LIMIT", 10 ** 9))
    for i, d in list(cand.items())[:lim]:
        if len(d) < 2:
            continue
        frame, prop, site = tf._state(C, i, sim, lambda s: cf.Monitor.frame(fr, s))
        vis0, prop0 = mon.start(frame, prop)
        names = list(d); tgt = np.r_[C["target"][i], 0.0]
        plans = np.stack([np.concatenate([tgt[None], d[a][0], np.repeat(d[a][0][-1:], 7, 0)]).astype(np.float32) for a in names])
        plans[:, 0, 3] = plans[:, 1, 3]
        keep = mon.hand_keep(site, plans[0])                     # one mask per decision (common to all chunks)
        r = mon.score(vis0, prop0, plans, keep, n=64)
        for j, a in enumerate(names):
            rows.append({"i": i, "arm": a, "risk": d[a][1], "h": float(r["h"][j]), "g": float(r["g"][j]), "o": float(r["o"][j])})
        if i % 10 == 0:
            print(name, i, len(rows), flush=True)
    (D / f"tokrank_{name}_{tag}_rows.json").write_text(json.dumps(rows))
    analyse(name, tag)


def conc(rows, key, sign):
    c = dcn = t = 0
    for i in {r["i"] for r in rows}:
        rs = [r for r in rows if r["i"] == i]
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                dr = rs[a]["risk"] - rs[b]["risk"]
                if abs(dr) < 2 / 32 - 1e-9:
                    continue
                ds = sign * (rs[a][key] - rs[b][key])
                c, dcn, t = c + (ds * dr > 0), dcn + (ds * dr < 0), t + (ds * dr == 0)
    return {"concordant": int(c), "discordant": int(dcn), "tied": int(t), "conc": round(c / max(c + dcn, 1), 3)}


def analyse(name, tag):
    from wm_libero import auroc
    T = json.loads((D / f"tokrank_{name}_{tag}_rows.json").read_text())
    O = {(r["i"], r["arm"]): r for r in json.loads((D / f"rank_{name}_rows.json").read_text())}
    T = [r for r in T if (r["i"], r["arm"]) in O]; Ob = [O[(r["i"], r["arm"])] for r in T]       # same chunks
    y = np.array([r["risk"] >= 2 / 32 for r in T])
    out = {"set": name, "model": tag, "chunks": len(T), "decisions": len({r["i"] for r in T}),
           "token_wm": {k: conc(T, k, s) for k, s in (("h", -1), ("g", 1), ("o", 1))},
           "model_OIL": {k: conc(Ob, k, s) for k, s in (("h", -1), ("o", 1))},
           "pooled_auroc": {"tok_minus_h": auroc(-np.array([r["h"] for r in T]), y), "tok_g": auroc(np.array([r["g"] for r in T]), y),
                            "tok_o": auroc(np.array([r["o"] for r in T]), y),
                            "oil_minus_h": auroc(-np.array([r["h"] for r in Ob]), y), "oil_o": auroc(np.array([r["o"] for r in Ob]), y)},
           "fork_rate": {"tok": float(np.mean([r["h"] < np.log2(3) - 1e-9 for r in T])),
                         "oil": float(np.mean([r["h"] < np.log2(3) - 1e-9 for r in Ob]))}}
    (D / f"tokrank_{name}_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4] == "analyse":
        analyse(sys.argv[1], sys.argv[3])
    else:
        main(*sys.argv[1:4])
