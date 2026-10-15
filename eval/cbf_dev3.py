"""Analysis of the dev3 CBF variant comparison (protocol results/jenga/cbf_dev3_protocol.json).

usage: python3 eval/cbf_dev3.py
"""
import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"A": "paired_cbf_dev3A", "B1": "paired_cbf_dev3B1", "B2": "paired_cbf_dev3B2"}


def load(tag):
    p = ROOT / f"results/jenga/{tag}.jsonl"
    return {(r := json.loads(l))["episode"]: r for l in p.read_text().splitlines()} if p.exists() else {}


top = lambda r: bool(r["neighbour_toppled"])
succ = lambda r: r.get("seconds_to_3cm") is not None and not top(r)


def main():
    base = load("paired_record_dev3")
    out = {"episodes_record": len(base), "arms": {}}
    for name, tag in ARMS.items():
        f = load(tag)
        E = sorted(set(base) & set(f))
        if not E:
            continue
        b = sum(top(base[e]) and not top(f[e]) for e in E); c = sum(top(f[e]) and not top(base[e]) for e in E)
        L = [x for e in E for x in f[e]["log"]]
        out["arms"][name] = {"episodes": len(E), "topples_alone": sum(top(base[e]) for e in E), "topples": sum(top(f[e]) for e in E),
                             "prevented": b, "caused": c,
                             "p_one_sided": round(binomtest(b, b + c, 0.5, alternative="greater").pvalue, 4) if b + c else None,
                             "success_alone": sum(succ(base[e]) for e in E), "success": sum(succ(f[e]) for e in E),
                             "interventions": sum(bool(x.get("intervened")) for x in L), "decisions": len(L)}
    A = out["arms"].get("A")
    if A and all(k in out["arms"] for k in ARMS) and len({v["episodes"] for v in out["arms"].values()}) == 1:
        ok = {k: v for k, v in out["arms"].items() if v["success"] >= A["success"] - 5}
        best = min(ok, key=lambda k: (ok[k]["topples"], k != "A"))
        out["choice"] = best if best == "A" or ok[best]["topples"] <= A["topples"] - 3 else "A"
    (ROOT / "results/jenga/cbf_dev3_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
