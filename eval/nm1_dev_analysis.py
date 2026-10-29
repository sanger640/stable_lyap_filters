"""NM1 dev analysis as pre-registered (results/jenga/nm1_protocol.json), written before any NM1 result.
No-mask value (value_irrnm8_alea.pt) at the 90% (nm10) and 80% (nm20) barriers, paired with the existing masked-value dev
audits g10 / g20 on dev 15000-15099, 2x and 1x.
usage: python3 eval/nm1_dev_analysis.py -> results/jenga/nm1_dev_result.json"""
import json
from math import comb
from pathlib import Path

import abl1_analysis as A

ROOT = Path(__file__).resolve().parent.parent
A.ARMS = {"nm10": "paired_lfc000_nm10dev{n}", "nm20": "paired_lfc000_nm20dev{n}",
          "g10": "paired_lfc000_g10dev{n}", "g20": "paired_lfc000_g20dev{n}"}
PAIRS = (("nm10", "g10"), ("nm20", "g20"))
KEYS = ("topple", "v4_success", "v4b_success", "red_fall", "timeout")


def main():
    R = {nz: {a: A.rows(a, nz) for a in A.ARMS} for nz in ("2x", "1x")}
    out = {"counts": {nz: {a: A.counts(v) for a, v in R[nz].items() if v} for nz in R},
           "pairs": {nz: {f"{a}_vs_{b}": {k: A.paired(R[nz][a], R[nz][b], k) for k in KEYS}
                          for a, b in PAIRS if R[nz][a] and R[nz][b]} for nz in R}}
    C, P = out["counts"], out["pairs"]
    done = all(f"{a}_vs_{b}" in P[nz] for nz in P for a, b in PAIRS)
    out["complete"] = done
    if done:
        # H-red: fewer red-block falls before extraction, pooled over both barriers and both noise levels (one-sided exact McNemar)
        fewer = sum(P[nz][f"{a}_vs_{b}"]["red_fall"]["b_only"] for nz in P for a, b in PAIRS)
        more = sum(P[nz][f"{a}_vs_{b}"]["red_fall"]["a_only"] for nz in P for a, b in PAIRS)
        n = fewer + more
        p = sum(comb(n, i) for i in range(fewer, n + 1)) / 2 ** n if n else 1.0
        out["H_red"] = {"nm_only_red_falls": more, "masked_only_red_falls": fewer, "p_one_sided_nm_fewer": float(f"{p:.3g}"),
                        "pass": bool(p < 0.05)}
        # H-same: per barrier, 2x topples <= masked + 3; v4 pooled over 2x and 1x >= masked - 3; 1x no cost (v4 >= 80, topples <= 5)
        hs = {}
        for a, b in PAIRS:
            t_ok = C["2x"][a]["topple"] <= C["2x"][b]["topple"] + 3
            v_ok = C["2x"][a]["v4_success"] + C["1x"][a]["v4_success"] >= C["2x"][b]["v4_success"] + C["1x"][b]["v4_success"] - 3
            c_ok = C["1x"][a]["v4_success"] >= 80 and C["1x"][a]["topple"] <= 5
            hs[a] = {"topples_2x_ok": t_ok, "v4_pooled_ok": v_ok, "no_cost_1x": c_ok, "pass": bool(t_ok and v_ok and c_ok)}
        out["H_same"] = hs
        out["GO_fresh"] = bool(out["H_red"]["pass"] and all(h["pass"] for h in hs.values()))
    (ROOT / "results/jenga/nm1_dev_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
