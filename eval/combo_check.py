"""Exploratory (dev sets only: ocl, m3): can the world-model alarm beat relative total variance (tvar30_rel_s4.0) alone?
Rules at the set's fork_3x safe alarm rate: tv alone; tv OR o > O_MAX; fork_3x OR tv; tv at other scales / horizons;
sum of tv10 and tv30. Metric: topple episodes warned >= 3 chunks ahead (8-chunk window).
usage: python3 eval/combo_check.py -> results/jenga/deep_dive/combo_check.json"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import seed_pool_check as sp  # noqa: E402

D = ROOT / "results/jenga/deep_dive"; RUNS = ROOT / "results/program_b/policy_rollouts"
SETS = {"ocl": (D / "spread_rows.jsonl", D, "paired_record_ocl"), "m3": (D / "spread_rows_m3.jsonl", D / "m3", "paired_record_m3")}
OM = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]


def one(name):
    srows, mdir, run = SETS[name]
    W = {(r["ep"], r["k"]): r for r in map(json.loads, srows.read_text().splitlines())}
    hi = srows.with_name(srows.name.replace("spread_rows", "spread_rows_hi"))         # scales 6 / 8 / 12 (SP_SC), same seeds
    if hi.exists():
        for r in map(json.loads, hi.read_text().splitlines()):
            if (r["ep"], r["k"]) in W:
                W[(r["ep"], r["k"])]["stats"].update(r["stats"])
        W = {k: v for k, v in W.items() if "8.0" in v["stats"]}
    M = {(r["ep"], r["k"]): r for f in sorted(mdir.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    keys = sorted(set(W) & set(M))
    A = json.loads((ROOT / f"results/jenga/outcome_audit_{name}.json").read_text())[run]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((RUNS / run / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None) for e in {k[0] for k in keys}}
    safe = np.array([not top[k[0]] for k in keys])
    rel = lambda s, st: np.array([np.log(W[k]["stats"][s][st][0] + 1e-12) - np.log(W[k]["stats"][s][st][1] + 1e-12) for k in keys])
    o = np.array([M[k]["o"] for k in keys]) > OM
    fork = np.array([sp.h_of(M[k]["plan"]) < min(sp.h_of(M[k]["hold"]), 2.0) - 1e-9 for k in keys]) | o
    rate = float(fork[safe].mean())

    def lead(f):
        by = {}
        for (e, k), x in zip(keys, f):
            by.setdefault(e, {})[k] = bool(x)
        hit = []
        for e, ks in by.items():
            if top[e] and kt.get(e) is not None:
                h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                hit.append(bool(h) and kt[e] - h[0] >= 3)
        return round(float(np.mean(hit)), 3), len(hit)

    def at_rate(score, base=None):                   # threshold on score so that (base OR score > t) has the target rate
        base = np.zeros(len(keys), bool) if base is None else base
        qs = np.quantile(score[safe], np.linspace(0.5, 1, 501))
        best = min(qs, key=lambda t: abs(float((base | (score > t))[safe].mean()) - rate))
        f = base | (score > best); return {"rate": round(float(f[safe].mean()), 4), "lead_ge3": lead(f)[0]}
    tv30_4, tv10_4, tv30_3, tv30_2 = rel("4.0", "tvar30"), rel("4.0", "tvar10"), rel("3.0", "tvar30"), rel("2.0", "tvar30")
    out = {"set": name, "fork_3x_rate": round(rate, 4), "topple_episodes": lead(fork)[1], "fork_3x_lead": lead(fork)[0],
           "tv30_s4": at_rate(tv30_4), "tv30_s4_OR_o": at_rate(tv30_4, o), "fork3x_OR_tv30_s4": at_rate(tv30_4, fork),
           "tv10_s4": at_rate(tv10_4), "tv30_s3": at_rate(tv30_3), "tv30_s2": at_rate(tv30_2),
           "tv10+tv30_s4": at_rate(tv10_4 + tv30_4), "max_scale_tv30": at_rate(np.maximum.reduce([tv30_2, tv30_3, tv30_4]))}
    if "8.0" in W[keys[0]]["stats"]:
        for sc in ("6.0", "8.0", "12.0"):
            out[f"tv30_s{sc[:-2]}"] = at_rate(rel(sc, "tvar30")); out[f"tv10_s{sc[:-2]}"] = at_rate(rel(sc, "tvar10"))
        out["decisions_with_hi"] = len(keys)
    return out


if __name__ == "__main__":
    res = [one(n) for n in SETS]
    (D / "combo_check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
