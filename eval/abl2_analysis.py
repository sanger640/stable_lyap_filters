"""ABL2 / VIS1 / SEED1 / TAU2 analysis as pre-registered (results/jenga/abl2_protocol.json), written before any of their episodes ran.
Statistics as eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/abl2_analysis.py abl2|vis1|seed1|tau2 -> section of results/jenga/abl2_result.json"""
import json
from pathlib import Path
import sys

import numpy as np

import h2h5_analysis as H

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "results/jenga"
ABL2 = ("o90", "b50", "b80", "lv0", "h16", "frag", "nofb", "act", "proj", "det", "mean", "soft")
LOOKS = ("light", "colour", "camera")


def rows(audit, run):
    return {r["ep"]: r for r in json.loads((J / audit).read_text())[run]["rows"]}


def logs(run, eps):
    edits, sec = [], []
    for e in eps:
        d = json.loads((H.RUNS / run / f"ep{e}.json").read_text())
        edits.append(sum(c.get("lsf_overrides", 0) or 0 for c in d["log"])); sec += [c.get("seconds", 0) or 0 for c in d["log"]]
    return {"edits_per_episode": round(float(np.mean(edits)), 2), "compute_per_move_mean_s": round(float(np.mean(sec)), 3)}


def table(V, runs, ref_pairs):
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    out = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in V},
           "logs": {a: logs(runs[a], eps) for a in V if not runs[a].startswith("paired_record")}, "pairs": {}}
    for a, b in ref_pairs:
        out["pairs"][f"{a}_vs_{b}"] = {k: H.paired(V, eps, a, b, k) for k in H.KEYS}
    return out, eps


def abl2():
    runs = {"policy": "paired_record_h6_2x", "h6o90": "paired_lfc000_h6o90_2x", **{a: f"paired_lfc000_ab2_{a}" for a in ABL2}}
    V = {a: rows("outcome_audit_h6_2x.json", r) for a, r in runs.items() if a in ("policy", "h6o90")}
    V.update({a: rows("outcome_audit_ab2.json", runs[a]) for a in ABL2})
    V = {a: {e: r for e, r in v.items() if 26800 <= e < 27100} for a, v in V.items()}
    out, eps = table(V, runs, [(a, "o90") for a in ABL2[1:]] + [(a, "policy") for a in ABL2])
    same = sum(all(V["o90"][e][k] == V["h6o90"][e][k] for k in ("logged_neighbour_toppled", "v4_success", "v4b_success",
                                                                     "red_fell_before_extraction", "extracted")) for e in eps)
    out["o90_rerun_matches_h2h6"] = {"episodes_identical_outcomes": same, "of": len(eps)}
    return out


def vis1():
    runs = {"policy": "paired_record_h6_2x", "o90": "paired_lfc000_ab2_o90", **{lk: f"paired_lfc000_vis1_{lk}" for lk in LOOKS}}
    V = {"policy": rows("outcome_audit_h6_2x.json", runs["policy"]), "o90": rows("outcome_audit_ab2.json", runs["o90"])}
    V.update({lk: rows("outcome_audit_vis1.json", runs[lk]) for lk in LOOKS})
    V = {a: {e: r for e, r in v.items() if 26800 <= e < 27100} for a, v in V.items()}
    out, eps = table(V, runs, [(lk, "policy") for lk in LOOKS] + [(lk, "o90") for lk in LOOKS] + [("o90", "policy")])
    c = out["counts"]; out["reading"] = {}
    for lk in LOOKS:
        p = out["pairs"][f"{lk}_vs_policy"]; red = c["policy"]["topple"] - c["o90"]["topple"]
        out["reading"][lk] = {"works": bool(p["topple"]["upper95_pp"] < 0 and p["v4_success"]["lower95_pp"] >= -5),
                              "retention": round((c["policy"]["topple"] - c[lk]["topple"]) / red, 3) if red else None}
    off = J / "vis1_offline.json"
    if off.exists():
        out["offline"] = json.loads(off.read_text())
    return out


def seed1():
    runs = {"A": "paired_lfc000_x1a", "C": "paired_lfc000_x1c", "A2": "paired_lfc000_seed1_a2", "policy": "paired_record_x1"}
    V = {a: rows("outcome_audit_x1.json", runs[a]) for a in ("A", "C", "policy")}
    V["A2"] = rows("outcome_audit_seed1.json", runs["A2"])
    out, eps = table(V, runs, [("C", "A2"), ("A2", "A"), ("C", "A"), ("A2", "policy")])
    out["reading"] = {"C_gain_beyond_seed": bool(out["pairs"]["C_vs_A2"]["v4_success"]["lower95_pp"] > 0)}
    return out


def tau2():
    runs = {"policy": "paired_record_h6_2x", "o90": "paired_lfc000_ab2_o90", "t085": "paired_lfc000_tau2_t085"}
    V = {"policy": rows("outcome_audit_h6_2x.json", runs["policy"]), "o90": rows("outcome_audit_ab2.json", runs["o90"]),
         "t085": rows("outcome_audit_tau2.json", runs["t085"])}
    V = {a: {e: r for e, r in v.items() if 26800 <= e < 27100} for a, v in V.items()}
    out, _ = table(V, runs, [("t085", "o90"), ("t085", "policy")])
    return out


def main(stage):
    f = J / "abl2_result.json"
    res = json.loads(f.read_text()) if f.exists() else {"protocol": "results/jenga/abl2_protocol.json"}
    res[stage] = {"abl2": abl2, "vis1": vis1, "seed1": seed1, "tau2": tau2}[stage]()
    f.write_text(json.dumps(res, indent=1))
    r = res[stage]; print(json.dumps({"counts": r["counts"], "logs": r["logs"], **{k: r[k] for k in ("reading", "o90_rerun_matches_h2h6") if k in r}}, indent=1))
    for name, p in r["pairs"].items():
        print(name, {k: (v["diff_pp"], v["lower95_pp"], v["upper95_pp"], v["mcnemar_p"]) for k, v in p.items()})


if __name__ == "__main__":
    main(sys.argv[1])
