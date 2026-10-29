"""XFER1 analysis as pre-registered (results/jenga/xfer1_protocol.json), written before any XFER1 result.
Shifted physics (neighbours x0.3 mass), 300 fresh episodes 30400-30699 at 2x, paired arms:
  policy; A = model_OIL + value_irr8_alea.pt (no adaptation); B = model_S2S + value_irr8_alea.pt (dynamics fine-tuned only);
  C = model_S2S + value_irr8_s2s_alea.pt (value and margin relabelled / retrained in model_S2S's imagination); filters = 90%.
usage: python3 eval/xfer1_analysis.py -> results/jenga/xfer1_result.json"""
import json
from pathlib import Path

import h2h5_analysis as H

ROOT = Path(__file__).resolve().parent.parent
H.ARMS = {"policy": "paired_record_x1{n}", "A": "paired_lfc000_x1a{n}", "B": "paired_lfc000_x1b{n}", "C": "paired_lfc000_x1c{n}"}


def load():
    A = json.loads((ROOT / "results/jenga/outcome_audit_x1.json").read_text())
    V = {a: {r["ep"]: r for r in A[run.format(n="")]["rows"]} for a, run in H.ARMS.items()}
    eps = sorted(set.intersection(*(set(v) for v in V.values())))
    return V, eps


def main():
    V, eps = load()
    out = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in H.ARMS}}
    out["pairs"] = {f"{a}_vs_{b}": {k: H.paired(V, eps, a, b, k) for k in H.KEYS}
                    for a, b in (("B", "C"), ("B", "A"), ("C", "A"), ("A", "policy"), ("B", "policy"), ("C", "policy"))}
    bc = out["pairs"]["B_vs_C"]
    out["H1_old_value_transfers"] = bool(bc["topple"]["upper95_pp"] <= 3 and bc["v4_success"]["lower95_pp"] >= -5)
    out["H1_detail"] = {"topple_upper95_pp": bc["topple"]["upper95_pp"], "v4_lower95_pp": bc["v4_success"]["lower95_pp"]}
    (ROOT / "results/jenga/xfer1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
