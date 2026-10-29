"""RNG1 stage D analysis as pre-registered (results/jenga/rng1_protocol.json), written before any RNG1 result.
Dev episodes 15000-15099, the RNG1 value (value_rng8_alea.pt) at the 80% / 90% barrier (r20 / r10) paired with the
existing irreversibility-value dev audits g20 / g10. usage: python3 eval/rng1_dev_analysis.py -> results/jenga/rng1_dev_result.json"""
import json
from pathlib import Path

import abl1_analysis as A

ROOT = Path(__file__).resolve().parent.parent
A.ARMS = {"r20": "paired_lfc000_r20dev{n}", "r10": "paired_lfc000_r10dev{n}", "g20": "paired_lfc000_g20dev{n}", "g10": "paired_lfc000_g10dev{n}"}
PAIR = {"r20": "g20", "r10": "g10"}


def main():
    out = {"counts": {}, "pairs": {}}
    for nz in ("2x", "1x"):
        R = {a: A.rows(a, nz) for a in A.ARMS}
        out["counts"][nz] = {a: A.counts(v) for a, v in R.items() if v}
        out["pairs"][nz] = {f"{a}_vs_{b}": {k: A.paired(R[a], R[b], k) for k in ("topple", "v4_success", "v4b_success")}
                            for a, b in PAIR.items() if R[a] and R[b]}
    C = out["counts"]
    ok = lambda a: (a in C["1x"] and a in C["2x"] and C["1x"][a]["v4_success"] >= 80 and C["1x"][a]["topple"] <= 5
                    and C["2x"][a]["topple"] <= C["2x"][PAIR[a]]["topple"] and C["2x"][a]["v4_success"] >= C["2x"][PAIR[a]]["v4_success"] - 5)
    out["D_pass_per_barrier"] = {a: bool(ok(a)) for a in PAIR}
    out["D_pass"] = any(out["D_pass_per_barrier"].values())
    (ROOT / "results/jenga/rng1_dev_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
