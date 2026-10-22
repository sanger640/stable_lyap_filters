"""Pre-registered analysis of the pillar closed-loop filter test (docs/PILLAR_TASK.md "Closed-loop filter").
usage: python3 eval/pillar_filter_analysis.py -> results/pillar/filter_result.json"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "libero")]
D = ROOT / "results/pillar/filter"
SEEDS = range(5000, 5300)


def load(arm):
    return {s: json.loads((D / arm / f"s{s}.json").read_text()) for s in SEEDS if (D / arm / f"s{s}.json").exists()}


def fail(r):
    return bool(r["outcome"]["cube_fell"] or r["outcome"]["pillar_tipped"])


def mcnemar(a, b, one_sided):
    """a, b bool arrays (event in arm A, arm B); one-sided H1: A has fewer events than B."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    if n10 + n01 == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    p = binomtest(n10, n10 + n01, 0.5, alternative="less" if one_sided else "two-sided").pvalue
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def boot_lower(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = a.astype(float) - b.astype(float)
    m = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return float(d.mean() * 100), float(np.percentile(m, 5) * 100)


def main():
    from wm_libero import auroc
    R = {a: load(a) for a in ("record", "hold", "g")}
    common = sorted(set(R["record"]) & set(R["hold"]) & set(R["g"]))
    F = {a: np.array([fail(R[a][s]) for s in common]) for a in R}
    S = {a: np.array([R[a][s]["outcome"]["success"] for s in common]) for a in R}
    out = {"episodes": len(common), "failures": {a: int(F[a].sum()) for a in R}, "success": {a: int(S[a].sum()) for a in R},
           "cube_fell": {a: int(sum(R[a][s]["outcome"]["cube_fell"] for s in common)) for a in R},
           "pillar_tipped": {a: int(sum(R[a][s]["outcome"]["pillar_tipped"] for s in common)) for a in R}}
    out["H1_g_vs_record_fail"] = mcnemar(F["g"], F["record"], True)
    diff, lo = boot_lower(S["g"], S["record"])
    out["H2_g_vs_record_success"] = {"diff_pp": diff, "lower95_pp": lo}
    out["H1_pass"] = out["H1_g_vs_record_fail"]["p"] < 0.05
    out["H2_pass"] = lo > -5
    out["universality"] = bool(out["H1_pass"] and out["H2_pass"])
    hd, hl = boot_lower(S["hold"], S["record"])
    out["secondary"] = {"hold_vs_record_fail": mcnemar(F["hold"], F["record"], True),
                        "hold_vs_record_success": {"diff_pp": hd, "lower95_pp": hl},
                        "g_vs_hold_fail": mcnemar(F["g"], F["hold"], False),
                        "g_vs_hold_success": mcnemar(~S["g"], ~S["hold"], False)}
    for a in R:
        L = [r for s in common for r in R[a][s]["log"]]
        kinds = {}
        for r in L:
            if r.get("kind"):
                kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        out["secondary"][f"{a}_decisions"] = len(L)
        out["secondary"][f"{a}_trigger_rate"] = float(np.mean([r["trig"] for r in L]))
        out["secondary"][f"{a}_kinds"] = kinds
        out["secondary"][f"{a}_median_sec"] = float(np.median([r["sec"] for r in L]))
    th = np.array([r["theta"] for s in common for r in R["g"][s]["log"] if r.get("kind") == "certified"])
    if len(th):
        out["secondary"]["g_theta_median"] = [round(float(x), 3) for x in np.median(th, 0)]
    dmax = np.array([max(r["d"] for r in R["record"][s]["log"]) for s in common])
    out["secondary"]["record_auroc_maxd_vs_fail"] = auroc(dmax, F["record"])
    trig = np.array([any(r["trig"] for r in R["record"][s]["log"]) for s in common])
    out["secondary"]["record_fail_rate_triggered_vs_not"] = [float(F["record"][trig].mean()) if trig.any() else None,
                                                             float(F["record"][~trig].mean()) if (~trig).any() else None]
    (ROOT / "results/pillar/filter_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
