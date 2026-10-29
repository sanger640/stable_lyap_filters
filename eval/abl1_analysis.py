"""ABL1 analysis as pre-registered (results/jenga/abl1_protocol.json), written before any ABL1 episode.
Dev episodes 15000-15099; paired with the existing g20 / g10 dev audits. usage: python3 eval/abl1_analysis.py -> results/jenga/abl1_result.json"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"lv0": "paired_lfc000_lv0dev{n}", "lv30": "paired_lfc030_lv30dev{n}", "lsr30": "paired_lfa030_lsr30dev{n}",
        "a16g20": "paired_lfc000_a16g20dev{n}", "g20": "paired_lfc000_g20dev{n}", "g10": "paired_lfc000_g10dev{n}"}


def rows(arm, nz):
    tag = ARMS[arm].format(n=nz).split("_", 2)[2]
    f = ROOT / f"results/jenga/outcome_audit_{tag}.json"
    if not f.exists():
        return None
    return {r["ep"]: r for r in json.loads(f.read_text())[ARMS[arm].format(n=nz)]["rows"]}


def outcome(r, key):
    if key == "topple":
        return bool(r["logged_neighbour_toppled"])
    if key == "timeout":
        return not r["extracted"] and not r["logged_neighbour_toppled"] and not r["red_fell_before_extraction"]
    if key == "red_fall":
        return bool(r["red_fell_before_extraction"])
    return bool(r[key])


def counts(R):
    return {k: sum(outcome(r, k) for r in R.values()) for k in ("topple", "v4_success", "v4b_success", "red_fall", "timeout")}


def paired(A, B, key):
    eps = sorted(set(A) & set(B))
    a10 = sum(outcome(A[e], key) and not outcome(B[e], key) for e in eps); a01 = sum(not outcome(A[e], key) and outcome(B[e], key) for e in eps)
    n = a10 + a01
    p_more = sum(comb(n, i) for i in range(a10, n + 1)) / 2 ** n if n else 1.0       # one-sided: A has MORE events than B
    return {"episodes": len(eps), "a_only": a10, "b_only": a01, "p_one_sided_a_more": float(f"{p_more:.3g}")}


def main():
    out = {"counts": {}, "pairs": {}}
    for nz in ("2x", "1x"):
        R = {a: rows(a, nz) for a in ARMS}
        out["counts"][nz] = {a: counts(v) for a, v in R.items() if v}
        out["pairs"][nz] = {f"{a}_vs_{b}": {k: paired(R[a], R[b], k) for k in ("topple", "v4_success")}
                            for a in ("lv0", "lv30", "lsr30", "a16g20") for b in ("g20", "g10") if R[a] and R[b]}
    no_cost = lambda a: (lambda c: c["v4_success"] >= 80 and c["topple"] <= 5)(out["counts"]["1x"][a]) if a in out["counts"]["1x"] else None
    worse = lambda a: out["pairs"]["2x"].get(f"{a}_vs_g10", {}).get("topple", {}).get("p_one_sided_a_more", 1.0) < 0.05
    lv = [a for a in ("lv0", "lv30", "lsr30") if a in out["counts"]["2x"] and a in out["counts"]["1x"]]
    out["no_cost_1x"] = {a: no_cost(a) for a in out["counts"]["1x"]}
    out["Q1_cbf_needed"] = bool(len(lv) == 3 and all(worse(a) or not no_cost(a) for a in lv)) if len(lv) == 3 else None
    out["Q1_per_arm"] = {a: {"more_2x_topples_than_g10_p<0.05": worse(a), "no_cost_1x": no_cost(a)} for a in lv}
    (ROOT / "results/jenga/abl1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
