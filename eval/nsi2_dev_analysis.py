"""NSI2 dev analysis as pre-registered (results/jenga/nsi2_protocol.json), written before any NSI2 or RNG1 result.
n20 / n10 (value_rngnsi8_alea.pt) paired with RNG1's r20 / r10, else with g20 / g10 (descriptive).
usage: python3 eval/nsi2_dev_analysis.py -> results/jenga/nsi2_dev_result.json"""
import json
from pathlib import Path

import abl1_analysis as A

ROOT = Path(__file__).resolve().parent.parent
A.ARMS = {"n20": "paired_lfc000_n20dev{n}", "n10": "paired_lfc000_n10dev{n}", "r20": "paired_lfc000_r20dev{n}", "r10": "paired_lfc000_r10dev{n}",
          "g20": "paired_lfc000_g20dev{n}", "g10": "paired_lfc000_g10dev{n}"}


def main():
    R = {nz: {a: A.rows(a, nz) for a in A.ARMS} for nz in ("2x", "1x")}
    rng_ok = all(R[nz][a] for nz in R for a in ("r20", "r10"))
    PAIR = {"n20": "r20", "n10": "r10"} if rng_ok else {"n20": "g20", "n10": "g10"}
    out = {"comparator": "RNG1 r20 / r10" if rng_ok else "g20 / g10 (descriptive; RNG1 dev arms missing)",
           "counts": {nz: {a: A.counts(v) for a, v in R[nz].items() if v} for nz in R},
           "pairs": {nz: {f"{a}_vs_{b}": {k: A.paired(R[nz][a], R[nz][b], k) for k in ("topple", "v4_success", "v4b_success")}
                          for a, b in PAIR.items() if R[nz][a] and R[nz][b]} for nz in R}}
    C = out["counts"]
    ok = lambda a: (a in C["2x"] and a in C["1x"] and C["2x"][a]["topple"] <= C["2x"][PAIR[a]]["topple"] + 3
                    and C["2x"][a]["v4_success"] >= C["2x"][PAIR[a]]["v4_success"] - 5 and C["1x"][a]["v4_success"] >= 80 and C["1x"][a]["topple"] <= 5)
    out["per_barrier"] = {a: bool(ok(a)) for a in PAIR}
    out["costs_nothing"] = bool(rng_ok and all(out["per_barrier"].values()))
    (ROOT / "results/jenga/nsi2_dev_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
