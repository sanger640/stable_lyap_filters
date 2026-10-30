"""GFCL1 analysis as pre-registered (results/jenga/gfcl1_protocol.json), written before any GFCL1 episode exists.
nm90 (no-mask value) and ns90 (0.8 s irreversible-fragility value), both at the 90% barrier, on H2H5's episodes 26200-26799,
paired with H2H5's ours90, policy and UNISafe arms (audits outcome_audit_h5_{2x,1x}.json; new: outcome_audit_h5{nm,ns}_{2x,1x}.json).
Statistics as eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/gfcl1_analysis.py -> results/jenga/gfcl1_result.json"""
import json
from math import comb
from pathlib import Path

import numpy as np

import h2h5_analysis as H

ROOT = Path(__file__).resolve().parent.parent
H.ARMS = {"policy": "paired_record_h5_{n}", "unisafe": "paired_lsfa030_h5_{n}", "ours90": "paired_lfc000_h5o90_{n}",
          "nm90": "paired_lfc000_h5nm90_{n}", "ns90": "paired_lfc000_h5ns90_{n}"}


def load(nz):
    A = json.loads((ROOT / f"results/jenga/outcome_audit_h5_{nz}.json").read_text())
    for t in ("nm", "ns"):
        A.update(json.loads((ROOT / f"results/jenga/outcome_audit_h5{t}_{nz}.json").read_text()))
    V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in H.ARMS.items()}
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    return V, eps


def analyse(nz):
    V, eps = load(nz)
    out = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in H.ARMS}, "time": {}}
    for a in H.ARMS:
        succ = [e for e, ok in zip(eps, H.vec(V, eps, a, "v4_success")) if ok]
        dur = [json.loads((H.RUNS / H.ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["chunks"] * 0.8 for e in succ]
        out["time"][a] = {"successes": len(succ), "mean_episode_duration_s": round(float(np.mean(dur)), 2) if dur else None}
    out["pairs"] = {f"{a}_vs_{b}": {k: H.paired(V, eps, a, b, k) for k in H.KEYS}
                    for a in ("nm90", "ns90") for b in ("ours90", "policy", "unisafe")}
    return out, V, eps


def pooled_red(arm, data):
    """one-sided exact McNemar over 2x and 1x pairs: arm has fewer red falls before extraction than ours90"""
    n10 = n01 = 0
    for V, eps in data:
        x, y = H.vec(V, eps, arm, "red_fall"), H.vec(V, eps, "ours90", "red_fall")
        n10 += int(((x == 1) & (y == 0)).sum()); n01 += int(((x == 0) & (y == 1)).sum())
    n = n10 + n01
    p1 = sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n if n else 1.0
    return {"pairs": sum(len(e) for _, e in data), "arm_only": n10, "ours90_only": n01, "p_one_sided_arm_fewer": float(f"{p1:.3g}")}


def main():
    (r2, V2, e2), (r1, V1, e1) = analyse("2x"), analyse("1x")
    p = lambda r, a, k: r["pairs"][f"{a}_vs_ours90"][k]
    noninf = lambda r, a: bool(p(r, a, "topple")["upper95_pp"] < 3 and p(r, a, "v4_success")["lower95_pp"] > -5)
    red = {a: pooled_red(a, [(V2, e2), (V1, e1)]) for a in ("nm90", "ns90")}
    hyp = {"R_nm": red["nm90"]["p_one_sided_arm_fewer"] < 0.05,
           "S_nm_2x": noninf(r2, "nm90"), "S_nm_1x": noninf(r1, "nm90"),
           "S_ns_2x": noninf(r2, "ns90"), "S_ns_1x": noninf(r1, "ns90")}
    out = {"stage": "gfcl1", "hypotheses": hyp, "red_pooled": red, "2x": r2, "1x": r1}
    (ROOT / "results/jenga/gfcl1_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"hypotheses": hyp, "red_pooled": red, "counts_2x": r2["counts"], "counts_1x": r1["counts"],
                      "time_2x": r2["time"], "time_1x": r1["time"],
                      "vs_ours90_2x": {a: r2["pairs"][f"{a}_vs_ours90"] for a in ("nm90", "ns90")},
                      "vs_ours90_1x": {a: r1["pairs"][f"{a}_vs_ours90"] for a in ("nm90", "ns90")}}, indent=1))


if __name__ == "__main__":
    main()
