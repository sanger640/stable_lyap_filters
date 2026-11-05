"""ACAL1's label-free barrier rule for values made after ACAL1 (KNOCK stage 5 / HZN1): gamma at which the value's CBF check
fails on the same share of the policy's own moves (RNG1's 3000 calibration decisions, same seeded draws) as ours90's check
(value_irr8_alea.pt, gamma 0.1). Reuses eval/acal1.py's run() unchanged (VALUES and the output folder replaced), and its
grade() rule: closest share on the 0.001 grid, ties -> middle; at 1x too few failing moves (< 30) -> 2x's gamma.
LCBFJ1: ACALX_WIDE=1 widens the grid to 0.99 (the labelled value's 1x share is not reached within ACAL1's 0.6).
usage: python3 eval/acal_extra.py <key> <value path or name in lsf_v1>  -> results/jenga/acalx_<key>.json (prints G2= / G1=)"""
import json
import os
import sys

import numpy as np

import acal1 as A


def main(key, value):
    A.VALUES = {"ours": A.REF, key: value}
    A.OUT = A.ROOT / "results/jenga" / f"acalx_{key}"; A.OUT.mkdir(exist_ok=True)
    if not (A.OUT / "acal1_rows.npz").exists():
        A.run()
    D = np.load(A.OUT / "acal1_rows.npz"); out = {"key": key, "value": value, "decisions": int(len(D["ep"])), "scales": {}, "wide_grid": os.environ.get("ACALX_WIDE") == "1"}
    for s in A.SCALES:
        tgt = A.fail_share(D[f"ours_now_{s:g}"], D[f"ours_min_{s:g}"], A.G_REF)
        vn, vm = D[f"{key}_now_{s:g}"], D[f"{key}_min_{s:g}"]
        grid = np.round(np.arange(0.01, 0.9901, 0.001), 3) if os.environ.get("ACALX_WIDE") == "1" else A.GRID
        F = np.array([A.fail_share(vn, vm, g_) for g_ in grid]); d = np.abs(F - tgt)
        gam = round(float(np.median(grid[d == d.min()])), 3)
        out["scales"][f"{s:g}x"] = {"target_fail_share": round(tgt, 4), "gamma": gam, "fail_share_at_gamma": round(A.fail_share(vn, vm, gam), 4),
                                    "fail_share_at_0.1": round(A.fail_share(vn, vm, 0.1), 4), "median_V_now": round(float(np.median(vn)), 4),
                                    "share_V_now_below0": round(float((vn < 0).mean()), 4)}
    if out["scales"]["1x"]["target_fail_share"] * out["decisions"] < 30:
        out["scales"]["1x"]["gamma_matched"] = out["scales"]["1x"]["gamma"]; out["scales"]["1x"]["gamma"] = out["scales"]["2x"]["gamma"]
        out["scales"]["1x"]["fallback_to_2x"] = True
    (A.ROOT / f"results/jenga/acalx_{key}.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
    print(f"G2={out['scales']['2x']['gamma']}"); print(f"G1={out['scales']['1x']['gamma']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
