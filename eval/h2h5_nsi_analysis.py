"""H2H5-NSI analysis as pre-registered (results/jenga/nsi_cl_protocol.json), written before the NSI value exists.
The irreversible-fragility arm (ours80 with value_nsi8_alea.pt) on H2H5's episodes 26200-26799, paired with H2H5's ours80
and policy arms (their audits: outcome_audit_h5_{2x,1x}.json; this arm's: outcome_audit_h5nsi_{2x,1x}.json).
Statistics as eval/h2h5_analysis.py (paired by episode, normal approximation, one-sided 95% bounds, exact McNemar).
usage: python3 eval/h2h5_nsi_analysis.py -> results/jenga/h2h5_nsi_result.json"""
import json
from pathlib import Path

import numpy as np

import h2h5_analysis as H

ROOT = Path(__file__).resolve().parent.parent
H.ARMS = {"policy": "paired_record_h5_{n}", "ours80": "paired_lfc000_h5o80_{n}", "nsi80": "paired_lfc000_h5n80_{n}"}


def load(nz):
    A = json.loads((ROOT / f"results/jenga/outcome_audit_h5_{nz}.json").read_text())
    A.update(json.loads((ROOT / f"results/jenga/outcome_audit_h5nsi_{nz}.json").read_text()))
    V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in H.ARMS.items()}
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    return V, eps


def analyse(nz):
    V, eps = load(nz)
    out = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in H.ARMS}, "time": {}}
    for a in H.ARMS:
        _, out["time"][a] = H.times(nz, a, eps)
        succ = [e for e, ok in zip(eps, H.vec(V, eps, a, "v4_success")) if ok]
        dur = [json.loads((H.RUNS / H.ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["chunks"] * 0.8 for e in succ]
        out["time"][a]["per_success"] = {"successes": len(succ), "mean_episode_duration_s": round(float(np.mean(dur)), 2) if dur else None}
    out["pairs"] = {f"nsi80_vs_{b}": {k: H.paired(V, eps, "nsi80", b, k) for k in H.KEYS} for b in ("ours80", "policy")}
    return out


def main():
    r2, r1 = analyse("2x"), analyse("1x")
    p = lambda r, b, k: r["pairs"][f"nsi80_vs_{b}"][k]
    noninf = lambda r: p(r, "ours80", "topple")["upper95_pp"] < 3 and p(r, "ours80", "v4_success")["lower95_pp"] > -5
    prim = {"N1_2x_noninferior_to_ours80": noninf(r2), "N2_1x_noninferior_to_ours80": noninf(r1)}
    prim["costs_nothing_on_jenga"] = prim["N1_2x_noninferior_to_ours80"] and prim["N2_1x_noninferior_to_ours80"]
    sec = {"H1_2x_vs_policy": p(r2, "policy", "topple")["p_one_sided_a_fewer"] < 0.05 and p(r2, "policy", "v4_success")["lower95_pp"] > 0,
           "H2_1x_no_cost_vs_policy": p(r1, "policy", "v4_success")["lower95_pp"] > -5 and p(r1, "policy", "topple")["upper95_pp"] < 3,
           "v4b_2x_vs_ours80_lower95_pp": p(r2, "ours80", "v4b_success")["lower95_pp"],
           "v4b_1x_vs_ours80_lower95_pp": p(r1, "ours80", "v4b_success")["lower95_pp"]}
    out = {"stage": "h2h5_nsi", "primary": {k: bool(v) for k, v in prim.items()}, "secondary": sec, "2x": r2, "1x": r1}
    (ROOT / "results/jenga/h2h5_nsi_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"primary": out["primary"], "secondary": sec, "counts_2x": r2["counts"], "counts_1x": r1["counts"],
                      "vs_ours80_2x": r2["pairs"]["nsi80_vs_ours80"], "vs_ours80_1x": r1["pairs"]["nsi80_vs_ours80"]}, indent=1))


if __name__ == "__main__":
    main()
