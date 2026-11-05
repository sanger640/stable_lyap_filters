"""HZN1 analysis (results/jenga/hzn1_protocol.json): on ACAL1's fresh Jenga episodes 27400-27999 (paired, replay-audited),
is irreversible fragility (fx, 1.6 s) comparable to plain irreversibility at the same horizon (irr16, 1.6 s), and do the
one-filter candidates (fx without the safety-actor fallback; irreversible fragility at 0.8 s) keep ours90's Jenga results?
Paired differences and non-inferiority as ACAL1 (eval/h2h5_analysis.paired: one-sided 95% bounds, topples upper < +3 pp
and v4 lower > -5 pp); arms that did not run are reported as not run.
usage: python3 eval/hzn1_analysis.py -> results/jenga/hzn1_result.json"""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import h2h5_analysis as H  # noqa: E402

OUT = ROOT / "results/jenga"
ARMS = {"ours90": "ac1o90", "fx": "ac1fxn", "irr16": "hz1i16", "fxnf": "hz1fxnf", "f8": "hz1f8"}


def rows(tag, nz):
    f = OUT / f"outcome_audit_{tag}_{nz}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()); return {r["ep"]: r for r in d[f"paired_lfc000_{tag}_{nz}"]["rows"]}


def cls(r):
    if r["logged_neighbour_toppled"]:
        return "topple"
    if r["red_fell_before_extraction"]:
        return "red_fall"
    if not r["extracted"]:
        return "timeout"
    return "v4_success" if r["v4_success"] else "other"


def main():
    res = {"protocol": "results/jenga/hzn1_protocol.json"}
    for nz in ("2x", "1x"):
        V = {a: rows(t, nz) for a, t in ARMS.items()}; V = {a: v for a, v in V.items() if v is not None and len(v) >= 600}
        eps = sorted(set.intersection(*(set(v) for v in V.values())))
        r = {"episodes": len(eps), "not_run": [a for a in ARMS if a not in V],
             "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in V}, "pairs": {}}
        for a, b in (("fx", "irr16"), ("irr16", "ours90"), ("fx", "ours90"), ("fxnf", "ours90"), ("fxnf", "fx"), ("f8", "ours90"), ("f8", "fx")):
            if a in V and b in V:
                p = {k: H.paired(V, eps, a, b, k) for k in H.KEYS}
                p["noninferior"] = bool(p["topple"]["upper95_pp"] < 3 and p["v4_success"]["lower95_pp"] > -5)
                p["discordant_outcomes"] = {f"{x}->{y}": n for (x, y), n in collections.Counter(
                    (cls(V[b][e]), cls(V[a][e])) for e in eps if cls(V[a][e]) != cls(V[b][e])).most_common()}
                r["pairs"][f"{a}_vs_{b}"] = p
        res[nz] = r
    hyp = lambda a, b: (bool(res["2x"]["pairs"][f"{a}_vs_{b}"]["noninferior"] and res["1x"]["pairs"][f"{a}_vs_{b}"]["noninferior"])
                        if f"{a}_vs_{b}" in res["2x"]["pairs"] and f"{a}_vs_{b}" in res["1x"]["pairs"] else "not run")
    res["H_comp"] = hyp("fx", "irr16")          # irreversible fragility costs nothing vs plain irreversibility at 1.6 s
    res["J_fxnf"] = hyp("fxnf", "ours90")       # one-filter candidates keep ours90's Jenga results
    res["J_f8"] = hyp("f8", "ours90")
    (OUT / "hzn1_result.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))
    for h in ("H_comp", "J_fxnf", "J_f8"):
        print(f"{h}={res[h] if isinstance(res[h], str) else int(res[h])}")


if __name__ == "__main__":
    main()
