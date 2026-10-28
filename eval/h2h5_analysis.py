"""H2H5 analysis as pre-registered (results/jenga/h2h5_protocol.json), written before any H2H5 episode ran.
Inputs: results/jenga/outcome_audit_h5_{2x,1x}.json (replay audits) + per-episode logs in
results/program_b/policy_rollouts/<run>/ep*.json (task time, compute per move, edits).
Statistics as eval/h2h4_analysis.py: paired by episode, normal approximation, one-sided 95% bounds, exact McNemar.
usage: python3 eval/h2h5_analysis.py -> results/jenga/h2h5_result.json"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"policy": "paired_record_h5_{n}", "ours90": "paired_lfc000_h5o90_{n}", "ours80": "paired_lfc000_h5o80_{n}",
        "unisafe": "paired_lsfa030_h5_{n}", "lsf_pub": "paired_lsfp020_h5_{n}"}
KEYS = ("topple", "v4_success", "v4b_success", "red_fall", "timeout")


def load(nz):
    A = json.loads((ROOT / f"results/jenga/outcome_audit_h5_{nz}.json").read_text())
    V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in ARMS.items()}
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    return V, eps


def vec(V, eps, a, key):
    R = [V[a][e] for e in eps]
    if key == "topple":
        return np.array([r["logged_neighbour_toppled"] for r in R], float)
    if key == "red_fall":
        return np.array([r["red_fell_before_extraction"] for r in R], float)
    if key == "timeout":
        return np.array([not r["extracted"] and not r["logged_neighbour_toppled"] and not r["red_fell_before_extraction"] for r in R], float)
    return np.array([r[key] for r in R], float)


def paired(V, eps, a, b, key):
    x, y = vec(V, eps, a, key), vec(V, eps, b, key); d = x - y; se = d.std(ddof=1) / np.sqrt(len(d))
    n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()); n = n10 + n01; k = min(n10, n01)
    p2 = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    p1 = sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n if n else 1.0                       # one-sided: a has fewer
    return {"diff_pp": round(100 * d.mean(), 2), "lower95_pp": round(100 * (d.mean() - 1.645 * se), 2),
            "upper95_pp": round(100 * (d.mean() + 1.645 * se), 2), "a_only": n10, "b_only": n01,
            "mcnemar_p": float(f"{p2:.3g}"), "p_one_sided_a_fewer": float(f"{p1:.3g}")}


def times(nz, a, eps):
    """task time (simulated) and compute per move from the episode logs"""
    lift, dur, sec, edits = {}, [], [], []
    for e in eps:
        d = json.loads((RUNS / ARMS[a].format(n=nz) / f"ep{e}.json").read_text())
        if d.get("seconds_to_3cm") is not None:
            lift[e] = d["seconds_to_3cm"]
        dur.append(d["chunks"] * 0.8)
        sec += [c.get("seconds", 0) or 0 for c in d["log"]]
        edits.append(sum(c.get("lsf_overrides", 0) or 0 for c in d["log"]))
    L = np.array(list(lift.values())) if lift else np.array([np.nan])
    return lift, {"lifted": len(lift), "time_to_lift_mean_s": round(float(np.nanmean(L)), 2), "time_to_lift_median_s": round(float(np.nanmedian(L)), 2),
                  "episode_duration_mean_s": round(float(np.mean(dur)), 2), "compute_per_move_mean_s": round(float(np.mean(sec)), 3),
                  "compute_per_move_p95_s": round(float(np.percentile(sec, 95)), 3), "edits_per_episode": round(float(np.mean(edits)), 2)}


def analyse(nz):
    V, eps = load(nz)
    out = {"episodes": len(eps), "counts": {a: {k: int(vec(V, eps, a, k).sum()) for k in KEYS} for a in ARMS}, "time": {}}
    lifts = {}
    for a in ARMS:
        lifts[a], out["time"][a] = times(nz, a, eps)
        succ = [e for e, ok in zip(eps, vec(V, eps, a, "v4_success")) if ok]              # average time per success
        dur = [json.loads((RUNS / ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["chunks"] * 0.8 for e in succ]
        lt = [lifts[a][e] for e in succ if e in lifts[a]]
        out["time"][a]["per_success"] = {"successes": len(succ), "mean_time_to_lift_s": round(float(np.mean(lt)), 2) if lt else None,
                                         "lifted_successes": len(lt), "mean_episode_duration_s": round(float(np.mean(dur)), 2) if dur else None}
    for a in ARMS:
        if a != "policy":
            both = sorted(set(lifts[a]) & set(lifts["policy"]))
            out["time"][a]["paired_lift_time_diff_vs_policy_median_s"] = round(float(np.median([lifts[a][e] - lifts["policy"][e] for e in both])), 2) if both else None
            out["time"][a]["paired_lift_episodes"] = len(both)
    out["pairs"] = {f"{a}_vs_{b}": {k: paired(V, eps, a, b, k) for k in KEYS}
                    for a in ("ours90", "ours80") for b in ("policy", "unisafe", "lsf_pub")}
    out["pairs"].update({f"{a}_vs_policy": {k: paired(V, eps, a, "policy", k) for k in KEYS} for a in ("unisafe", "lsf_pub")})
    return out


def main():
    r2, r1 = analyse("2x"), analyse("1x")
    p = lambda r, a, b, k: r["pairs"][f"{a}_vs_{b}"][k]
    PRI, SEC = "ours80", "ours90"                       # user 2026-10-28, before launch: primary = 80% barrier
    H = {"H1_2x_vs_policy": p(r2, PRI, "policy", "topple")["p_one_sided_a_fewer"] < 0.05 and p(r2, PRI, "policy", "v4_success")["lower95_pp"] > 0,
         "H2_1x_no_cost": p(r1, PRI, "policy", "v4_success")["lower95_pp"] > -5 and p(r1, PRI, "policy", "topple")["upper95_pp"] < 3,
         "H3_2x_vs_unisafe_noninferior": p(r2, PRI, "unisafe", "topple")["upper95_pp"] < 3 and p(r2, PRI, "unisafe", "v4_success")["lower95_pp"] > -5
                                         and p(r2, PRI, "unisafe", "v4b_success")["lower95_pp"] > -5,
         "H4_2x_vs_lsf_pub": p(r2, PRI, "lsf_pub", "topple")["p_one_sided_a_fewer"] < 0.05}
    sec = {f"{h}_{SEC}": v for h, v in {
        "H1": p(r2, SEC, "policy", "topple")["p_one_sided_a_fewer"] < 0.05 and p(r2, SEC, "policy", "v4_success")["lower95_pp"] > 0,
        "H2": p(r1, SEC, "policy", "v4_success")["lower95_pp"] > -5 and p(r1, SEC, "policy", "topple")["upper95_pp"] < 3,
        "H3": p(r2, SEC, "unisafe", "topple")["upper95_pp"] < 3 and p(r2, SEC, "unisafe", "v4_success")["lower95_pp"] > -5
              and p(r2, SEC, "unisafe", "v4b_success")["lower95_pp"] > -5,
        "H4": p(r2, SEC, "lsf_pub", "topple")["p_one_sided_a_fewer"] < 0.05}.items()}
    sec[f"{PRI}_superior_to_unisafe_topples_2x"] = p(r2, PRI, "unisafe", "topple")["p_one_sided_a_fewer"] < 0.05
    out = {"stage": "h2h5", "primary_arm": PRI, "primary": {k: bool(v) for k, v in H.items()}, "secondary": {k: bool(v) for k, v in sec.items()}, "2x": r2, "1x": r1}
    (ROOT / "results/jenga/h2h5_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"primary_arm": PRI, "primary": out["primary"], "secondary": out["secondary"], "counts_2x": r2["counts"], "counts_1x": r1["counts"],
                      "time_per_success_2x": {a: r2["time"][a]["per_success"] for a in ARMS},
                      "time_per_success_1x": {a: r1["time"][a]["per_success"] for a in ARMS}}, indent=1))


if __name__ == "__main__":
    main()
