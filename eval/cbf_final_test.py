"""Final CBF test driver (results/jenga/cbf_final_test_protocol.json).

  decide   apply the pre-registered rules to results/jenga/closedloop_train_eval.json -> world model + horizon
  analyse  paired comparison of paired_record_test2x (policy alone) vs paired_cbf_test2x (with filter)
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def decide():
    ev = json.loads((ROOT / "results/jenga/closedloop_train_eval.json").read_text())
    o, n = ev["old"]["H15"], ev["new"]["H15"]
    use_new = (n["split_caught_forks"] - o["split_caught_forks"] >= 0.15) and (n["split_false_on_safe"] - o["split_false_on_safe"] <= 0.05)
    chosen = ev["new" if use_new else "old"]
    horizon = 8 if (chosen["H8"]["auc_barrier"] or 0) - (chosen["H15"]["auc_barrier"] or 0) >= 0.03 else 15
    out = {"world_model": str(ROOT / "results/program_b/closedloop_v1/model/model_US.pt") if use_new else "threeblock_v3",
           "use_new": bool(use_new), "n_look": horizon, "evidence": ev}
    (ROOT / "results/jenga/cbf_final_choice.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "evidence"}), flush=True)


def analyse():
    from scipy.stats import binomtest
    load = lambda t: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()}
    a, f = load("paired_record_test2x"), load("paired_cbf_test2x")
    eps = sorted(set(a) & set(f))
    top = lambda r: bool(r["neighbour_toppled"]); succ = lambda r: bool(r.get("seconds_to_3cm") is not None and not r["neighbour_toppled"])
    b = sum(top(a[e]) and not top(f[e]) for e in eps); c = sum(not top(a[e]) and top(f[e]) for e in eps)
    p = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else 1.0
    s0, s1 = sum(succ(a[e]) for e in eps), sum(succ(f[e]) for e in eps)
    L = [x for e in eps for x in f[e]["log"]]
    res = {"episodes": len(eps), "topples_policy_alone": sum(top(a[e]) for e in eps), "topples_with_filter": sum(top(f[e]) for e in eps),
           "prevented": b, "caused": c, "mcnemar_p_one_sided": round(p, 4), "success_policy_alone": s0, "success_with_filter": s1,
           "decisions": len(L), "interventions": sum(bool(x.get("intervened")) for x in L),
           "best_effort": sum(x.get("kind") == "best_effort" for x in L),
           "decision_seconds_median": sorted(x["seconds"] for x in L)[len(L) // 2] if L else None}
    res["filter_works"] = bool(len(eps) >= 300 and p < 0.05 and s0 - s1 <= 30)
    (ROOT / "results/jenga/cbf_final_test_result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"decide": decide, "analyse": analyse}[sys.argv[1]]()
