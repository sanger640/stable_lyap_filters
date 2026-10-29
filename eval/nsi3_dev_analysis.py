"""NSI3 dev analysis as pre-registered (results/jenga/nsi3_protocol.json), written before any NSI3 result.
n16g20 (value_nsi16_alea.pt, 80% barrier) paired with ABL1's a16g20 on dev 15000-15099.
usage: python3 eval/nsi3_dev_analysis.py -> results/jenga/nsi3_dev_result.json"""
import json
from pathlib import Path

import abl1_analysis as A

ROOT = Path(__file__).resolve().parent.parent
A.ARMS = {"n16g20": "paired_lfc000_n16g20dev{n}", "a16g20": "paired_lfc000_a16g20dev{n}"}


def main():
    R = {nz: {a: A.rows(a, nz) for a in A.ARMS} for nz in ("2x", "1x")}
    out = {"counts": {nz: {a: A.counts(v) for a, v in R[nz].items() if v} for nz in R},
           "pairs": {nz: {k: A.paired(R[nz]["n16g20"], R[nz]["a16g20"], k) for k in ("topple", "v4_success", "v4b_success")}
                     for nz in R if R[nz]["n16g20"] and R[nz]["a16g20"]}}
    C = out["counts"]
    out["costs_nothing"] = bool(all("n16g20" in C[nz] for nz in C) and C["2x"]["n16g20"]["topple"] <= C["2x"]["a16g20"]["topple"] + 3
                                and C["2x"]["n16g20"]["v4_success"] >= C["2x"]["a16g20"]["v4_success"] - 5
                                and C["1x"]["n16g20"]["v4_success"] >= 80 and C["1x"]["n16g20"]["topple"] <= 5)
    (ROOT / "results/jenga/nsi3_dev_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
