"""ABL3 analysis as pre-registered (results/jenga/abl3_protocol.json), written before any ABL3 episode ran.
Statistics as eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/abl3_analysis.py -> results/jenga/abl3_result.json"""
import json

import numpy as np

import h2h5_analysis as H
from abl2_analysis import J, rows, table

ONE = ("o90", "b50", "b80", "lv0", "det", "mean", "g90", "gk")
MODE = {"g90": "lfcg090", "gk": "lfck000"}


def gate_share(run, eps):
    L = [c for e in eps for c in json.loads((H.RUNS / run / f"ep{e}.json").read_text())["log"]]
    return round(float(np.mean([bool(c.get("irr_gate")) for c in L])), 4)


def scale(nz):
    arms = ONE if nz == "1x" else ("g90", "gk")
    runs = {"policy": f"paired_record_h6_{nz}", **{a: f"paired_{MODE.get(a, 'lfc000')}_ab3_{a}_{nz}" for a in arms}}
    V = {"policy": rows(f"outcome_audit_h6_{nz}.json", runs["policy"])}
    V.update({a: rows(f"outcome_audit_ab3_{nz}.json", runs[a]) for a in arms})
    if nz == "2x":
        runs["o90"] = "paired_lfc000_ab2_o90"; V["o90"] = rows("outcome_audit_ab2.json", runs["o90"])
    else:
        runs["h6o90"] = "paired_lfc000_h6o90_1x"; V["h6o90"] = rows("outcome_audit_h6_1x.json", runs["h6o90"])
    V = {a: {e: r for e, r in v.items() if 26800 <= e < 27100} for a, v in V.items()}
    test = [a for a in arms if a != "o90"]
    out, eps = table(V, runs, [(a, "o90") for a in test] + [(a, "policy") for a in arms if a != "policy"] +
                     ([("o90", "policy")] if nz == "2x" else []))
    out["gate_open_share"] = {a: gate_share(runs[a], eps) for a in ("g90", "gk")}
    if nz == "1x":
        same = sum(all(V["o90"][e][k] == V["h6o90"][e][k] for k in ("logged_neighbour_toppled", "v4_success", "v4b_success",
                                                                     "red_fell_before_extraction", "extracted")) for e in eps)
        out["o90_rerun_matches_h2h6"] = {"episodes_identical_outcomes": same, "of": len(eps)}
    return out


if __name__ == "__main__":
    res = {"protocol": "results/jenga/abl3_protocol.json", "1x": scale("1x"), "2x": scale("2x")}
    (J / "abl3_result.json").write_text(json.dumps(res, indent=1))
    for nz in ("2x", "1x"):
        r = res[nz]
        print(nz, json.dumps({k: r[k] for k in ("counts", "logs", "gate_open_share", "o90_rerun_matches_h2h6") if k in r}, indent=1))
        for name, p in r["pairs"].items():
            print(nz, name, {k: (v["diff_pp"], v["lower95_pp"], v["upper95_pp"], v["mcnemar_p"]) for k, v in p.items()})
