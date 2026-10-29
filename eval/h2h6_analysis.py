"""H2H6 analysis as pre-registered (results/jenga/h2h6_protocol.json), written before any H2H6 episode ran.
Confirmatory fresh head-to-head on episodes 26800-27399: primary = ours90 (0.8 s irreversibility value, 90% barrier),
secondary = a16g20 (1.6 s value, 80% barrier); vs the policy alone and UNISafe. Statistics and outcome definitions as
eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/h2h6_analysis.py -> results/jenga/h2h6_result.json"""
import json
from pathlib import Path

import numpy as np

import h2h5_analysis as H

ROOT = Path(__file__).resolve().parent.parent
H.ARMS = {"policy": "paired_record_h6_{n}", "ours90": "paired_lfc000_h6o90_{n}", "a16g20": "paired_lfc000_h6a16_{n}", "unisafe": "paired_lsfa030_h6_{n}"}
KEYS = H.KEYS


def load(nz):
    A = json.loads((ROOT / f"results/jenga/outcome_audit_h6_{nz}.json").read_text())
    V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in H.ARMS.items()}
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    return V, eps


def analyse(nz):
    V, eps = load(nz)
    out = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in KEYS} for a in H.ARMS}, "time": {}}
    for a in H.ARMS:
        _, out["time"][a] = H.times(nz, a, eps)
        succ = [e for e, ok in zip(eps, H.vec(V, eps, a, "v4_success")) if ok]
        dur = [json.loads((H.RUNS / H.ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["chunks"] * 0.8 for e in succ]
        out["time"][a]["per_success"] = {"successes": len(succ), "mean_episode_duration_s": round(float(np.mean(dur)), 2) if dur else None}
    out["pairs"] = {f"{a}_vs_{b}": {k: H.paired(V, eps, a, b, k) for k in KEYS}
                    for a, b in (("ours90", "policy"), ("ours90", "unisafe"), ("a16g20", "policy"), ("a16g20", "unisafe"), ("a16g20", "ours90"), ("unisafe", "policy"))}
    return out


def hyp(r2, r1, arm):
    p = lambda r, b, k: r["pairs"][f"{arm}_vs_{b}"][k]
    return {"H1_2x_vs_policy": p(r2, "policy", "topple")["p_one_sided_a_fewer"] < 0.05 and p(r2, "policy", "v4_success")["lower95_pp"] > 0,
            "H2_1x_no_cost": p(r1, "policy", "v4_success")["lower95_pp"] > -5 and p(r1, "policy", "topple")["upper95_pp"] < 3,
            "H3_2x_vs_unisafe_noninferior": p(r2, "unisafe", "topple")["upper95_pp"] < 3 and p(r2, "unisafe", "v4_success")["lower95_pp"] > -5
                                            and p(r2, "unisafe", "v4b_success")["lower95_pp"] > -5,
            "S_2x_fewer_topples_than_unisafe": p(r2, "unisafe", "topple")["p_one_sided_a_fewer"] < 0.05}


def main():
    r2, r1 = analyse("2x"), analyse("1x")
    prim, sec = hyp(r2, r1, "ours90"), hyp(r2, r1, "a16g20")
    out = {"stage": "h2h6", "primary_arm": "ours90", "primary": {k: bool(v) for k, v in prim.items()},
           "primary_confirmed": bool(prim["H1_2x_vs_policy"] and prim["H2_1x_no_cost"] and prim["H3_2x_vs_unisafe_noninferior"]),
           "secondary_arm": "a16g20", "secondary": {k: bool(v) for k, v in sec.items()},
           "a16g20_vs_ours90_descriptive": {"2x": r2["pairs"]["a16g20_vs_ours90"], "1x": r1["pairs"]["a16g20_vs_ours90"]}, "2x": r2, "1x": r1}
    (ROOT / "results/jenga/h2h6_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("primary", "primary_confirmed", "secondary")} | {"counts_2x": r2["counts"], "counts_1x": r1["counts"],
                      "time_per_success_2x": {a: r2["time"][a]["per_success"] for a in H.ARMS}}, indent=1))


if __name__ == "__main__":
    main()
