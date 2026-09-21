"""Blind forks defined ACROSS seeds (PLAN_NEXT.md Phase 2, step 3).

One seed's misses mix what the dynamics cannot represent with what one optimisation run happened
to get wrong. This reads the frozen-benchmark evaluations of several seeds of one arm and, for every
physical fork state, counts

    q_i       = (# seeds whose monitor MISSES fork i at the pre-declared operating point) / seeds
    q_blind_i = (# seeds for which fork i is BLIND, score < blind_margin x threshold)   / seeds
    q_matched_i = (# seeds that miss fork i at a MATCHED 5% test false-positive rate) / seeds

The pre-declared operating point is set on development quiet states, so each seed realises a
different false-positive rate on test (2-6% across the gnn_n5 seeds against a 5% budget). A seed
whose threshold landed conservatively then "misses" forks for reasons unrelated to its dynamics.
q_matched puts every seed at the same test FPR; that threshold is chosen on TEST quiet states, so
it is an analysis equaliser, not a deployable operating point. A fork robust-blind under BOTH q and
q_matched is the strongest evidence of a systematic limitation.

Each is classified provisionally: robust blind (q_i >= 0.8), seed-sensitive (0.3 <= q_i < 0.8),
usually detected (q_i < 0.3). Raw counts are always kept, so the cutoffs can move later without
re-running the evaluation. Refuses to mix evaluations of different frozen benchmarks.
"""
import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROBUST, SENSITIVE = 0.8, 0.3
MATCHED = "5pct"


def classify(q):
    return "robust_blind" if q >= ROBUST else ("seed_sensitive" if q >= SENSITIVE
                                               else "usually_detected")


def main():
    ap = argparse.ArgumentParser()
    # Exact seed-file names only: a looser glob also matched the V1 evaluation
    # (w6_gnn_n5_s1_v1.json) and would have mixed vision scores into the privileged counts.
    ap.add_argument("--evals", nargs="+",
                    default=sorted(p for p in glob.glob(str(ROOT / "results/jenga/bench_eval/"
                                                            "w6_gnn_n5_s*.json"))
                                   if re.fullmatch(r"w6_gnn_n5_s\d+\.json", Path(p).name)))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/blind_freq_gnn_n5.json"))
    args = ap.parse_args()

    runs = [json.loads(Path(p).read_text()) for p in args.evals]
    hashes = {r["benchmark_sha256"] for r in runs}
    if len(hashes) != 1:
        raise SystemExit(f"evaluations come from different frozen benchmarks: {sorted(hashes)}")
    seeds = [r["model"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds: {seeds}")
    positive = runs[0]["eval_config"]["positive_class"]

    result = {"benchmark_sha256": hashes.pop(), "seeds": seeds, "evaluations": args.evals,
              "cutoffs": {"robust_blind": f"q >= {ROBUST}",
                          "seed_sensitive": f"{SENSITIVE} <= q < {ROBUST}",
                          "usually_detected": f"q < {SENSITIVE}"},
              "miss": "not alarmed at the pre-declared operating point (95th pct of dev quiet)",
              "scales": {}}
    for scale in map(str, runs[0]["eval_config"]["scales"]):
        forks = {}
        matched = {id(run): run["scales"][scale]["matched_fpr"][MATCHED]["threshold_mm"]
                   for run in runs}
        for run in runs:
            for r in run["rows"]:
                if r["split"] != "test" or r["scale"] != scale or r["class"] != positive:
                    continue
                key = (r["episode_id"], r["chunk_start"])
                entry = forks.setdefault(key, {"episode_id": key[0], "chunk_start": key[1],
                                               "topple_count": r["topple_count"],
                                               "misses": 0, "blinds": 0, "misses_matched": 0,
                                               "scores": {},
                                               "real_spread_mm": r["real_spread_mm"]})
                entry["misses"] += int(not r["alarm"])
                entry["misses_matched"] += int(not r["score"] > matched[id(run)])
                entry["blinds"] += int(r["blind"])
                entry["scores"][str(run["model"]["seed"])] = r["score"]
        n = len(runs)
        for entry in forks.values():
            if len(entry["scores"]) != n:
                raise SystemExit(f"fork {entry['episode_id']}:{entry['chunk_start']} missing "
                                 f"from some evaluations")
            entry["q"] = entry["misses"] / n
            entry["q_blind"] = entry["blinds"] / n
            entry["q_matched"] = entry["misses_matched"] / n
            entry["class"] = classify(entry["q"])
            entry["class_matched"] = classify(entry["q_matched"])
            entry["robust_under_both"] = (entry["class"] == "robust_blind"
                                          and entry["class_matched"] == "robust_blind")
            entry["median_score_mm"] = float(np.median(list(entry["scores"].values())))
        order = lambda e: (-e["q"], -e["q_blind"], e["episode_id"], e["chunk_start"])  # noqa: E731
        rows = sorted(forks.values(), key=order)
        q = np.array([e["q"] for e in rows])
        result["scales"][scale] = {
            "forks": len(rows), "seeds": n,
            "q_histogram": {f"{k}/{n}": int(v) for k, v in
                            sorted(Counter(e["misses"] for e in rows).items())},
            "q_blind_histogram": {f"{k}/{n}": int(v) for k, v in
                                  sorted(Counter(e["blinds"] for e in rows).items())},
            "q_matched_histogram": {f"{k}/{n}": int(v) for k, v in
                                    sorted(Counter(e["misses_matched"] for e in rows).items())},
            "classes": dict(Counter(e["class"] for e in rows)),
            "classes_matched": dict(Counter(e["class_matched"] for e in rows)),
            "robust_under_both": int(sum(e["robust_under_both"] for e in rows)),
            "realised_test_fp": {str(run["model"]["seed"]): run["scales"][scale]["pre_declared"]
                                 ["false_positive"]["rate"] for run in runs},
            "mean_q": float(q.mean()) if len(q) else None,
            "rows": rows}

    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(runs)} seeds {seeds}, benchmark {result['benchmark_sha256'][:16]}...")
    for scale, s in result["scales"].items():
        print(f"  {scale}x  {s['forks']} forks, mean q {s['mean_q']:.2f}, robust under both: "
              f"{s['robust_under_both']}")
        print(f"        pre-declared  classes {s['classes']}")
        print(f"                      misses -> forks {s['q_histogram']}")
        print(f"        matched 5%    classes {s['classes_matched']}")
        print(f"                      misses -> forks {s['q_matched_histogram']}")


if __name__ == "__main__":
    main()
