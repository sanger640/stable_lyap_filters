"""Decisive test B analysis (results/jenga/filter_decisive_protocol.json): paired closed-loop episodes, policy alone
(paired_monitor) vs policy + CEM filter (paired_cem, paired_cem_v2). McNemar exact test on neighbour topples; task success
= lift >= 3 cm with no neighbour toppled."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def load(tag):
    p = ROOT / f"results/jenga/{tag}.jsonl"
    return {r["episode"]: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {}


def main():
    from scipy.stats import binomtest
    base = load("paired_monitor"); out = {}
    for arm in ("paired_cem", "paired_cem_v2"):
        f = load(arm)
        eps = sorted(set(base) & set(f))
        if not eps:
            continue
        top = lambda r: bool(r["neighbour_toppled"])
        succ = lambda r: bool(r.get("seconds_to_3cm") is not None and not r["neighbour_toppled"])
        b = sum(top(base[e]) and not top(f[e]) for e in eps)      # filter prevented
        c = sum(not top(base[e]) and top(f[e]) for e in eps)      # filter caused
        p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        s0, s1 = sum(succ(base[e]) for e in eps), sum(succ(f[e]) for e in eps)
        L = [c_ for e in eps for c_ in f[e]["log"]]
        res = {"episodes": len(eps), "topples_policy_alone": sum(top(base[e]) for e in eps), "topples_with_filter": sum(top(f[e]) for e in eps),
               "prevented": b, "caused": c, "mcnemar_p_one_sided": round(float(p), 4),
               "success_policy_alone": s0, "success_with_filter": s1,
               "interventions": sum(bool(x.get("intervened")) for x in L), "holds": sum(bool(x.get("abstained")) for x in L),
               "decisions": len(L)}
        res["filter_works"] = bool(len(eps) >= 100 and p < 0.05 and (s0 - s1) <= 0.10 * len(eps))
        out[arm] = res
    (ROOT / "results/jenga/filter_decisive_B.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
