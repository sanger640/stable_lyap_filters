"""Regenerate every results table in the README from the raw per-trial JSONL files."""
import json
from math import comb
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def mcnemar(prevented, caused):
    """Exact two-sided McNemar on the discordant pairs."""
    n = prevented + caused
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(max(prevented, caused), n + 1)) / 2 ** n * 2)


def summarize(path, label, trust):
    rows = [json.loads(l) for l in open(path)]
    paired = {}
    for r in rows:
        paired.setdefault(r["seed"], {})[r["filtered"]] = r
    both = [v for v in paired.values() if len(v) == 2]
    prevented = sum(1 for v in both if v[False]["toppled"] and not v[True]["toppled"])
    caused = sum(1 for v in both if not v[False]["toppled"] and v[True]["toppled"])
    ctl = [r for r in rows if not r["filtered"]]
    trt = [r for r in rows if r["filtered"]]
    d0 = np.mean([r["div_before"] for r in trt if r["div_before"] is not None])
    d1 = np.mean([r["div_after"] for r in trt if r["div_after"] is not None])
    return dict(
        label=label, trust=trust, n_pairs=len(both),
        topple_ctl=np.mean([r["toppled"] for r in ctl]),
        topple_trt=np.mean([r["toppled"] for r in trt]),
        prevented=prevented, caused=caused, p=mcnemar(prevented, caused),
        corr_mm=np.mean([r["mean_abs_correction_m"] for r in trt]) * 1000,
        div_change=100 * (d0 - d1) / d0,
        pick_ctl=np.mean([r["picked"] for r in ctl]),
        pick_trt=np.mean([r["picked"] for r in trt]),
        err_ctl=np.mean([r["endpoint_err_m"] for r in ctl]),
        err_trt=np.mean([r["endpoint_err_m"] for r in trt]),
    )


def main():
    runs = [
        summarize(RESULTS / "active_filter_gradient_5mm.jsonl", "gradient (world model)", "5 mm"),
        summarize(RESULTS / "active_filter_random_5mm.jsonl", "random, matched magnitude", "5 mm"),
        summarize(RESULTS / "active_filter_gradient_1mm_buggyhist.jsonl",
                  "gradient (pre-fix history)", "1 mm"),
    ]

    print("| filter | trust | topple ctl -> filtered | prevented/caused | McNemar p | correction | predicted div | pick rate |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['label']} | {r['trust']} | {r['topple_ctl']:.1%} -> {r['topple_trt']:.1%} "
              f"| {r['prevented']}/{r['caused']} | {r['p']:.3f} | {r['corr_mm']:.2f} mm "
              f"| {r['div_change']:+.1f}% | {r['pick_ctl']:.0%} -> {r['pick_trt']:.0%} |")

    g, rnd = runs[0], runs[1]
    n = g["caused"] + rnd["caused"]
    p = sum(comb(n, k) for k in range(max(g["caused"], rnd["caused"]), n + 1)) / 2 ** n * 2
    print(f"\ngradient caused {g['caused']} topples vs random-direction {rnd['caused']}: "
          f"two-sided binomial p = {p:.3f}")
    print("-> the gradient direction is statistically indistinguishable from random noise of "
          "the same magnitude in its effect on real outcomes.")

    base = [json.loads(l) for l in open(RESULTS / "live_policy_base_rate.jsonl")]
    print(f"\nlive-policy base rate ({len(base)} trials): "
          f"topple {np.mean([r['toppled'] for r in base]):.1%}, "
          f"picked {np.mean([r['picked'] for r in base]):.1%}, "
          f"mean peak tilt {np.mean([r['peak_tilt_deg'] for r in base]):.2f} deg "
          f"(topple threshold 45 deg) -- the policy never grasps, so it never creates risk.")


if __name__ == "__main__":
    main()
