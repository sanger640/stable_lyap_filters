"""KNOCK stage 2 analysis (results/jenga/knock_protocol.json stage2): per-arm rates, paired differences vs the policy
(paired bootstrap), H_block and H_frag. usage: python3 eval/knock_analysis.py"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"policy": "paired_record_knock2", "ours90": "paired_lfc000_knock2_o90", "frag": "paired_lfc000_knock2_frag"}
EPS = range(45000, 45200)


def load(tag):
    out = {}
    for f in (RUNS / tag).glob("ep*.json"):
        r = json.loads(f.read_text()); out[int(r["episode"])] = r
    return out


def boot(x, y, seed=0, n=10_000):
    rng = np.random.default_rng(seed); d = x.astype(float) - y.astype(float); idx = rng.integers(len(d), size=(n, len(d)))
    m = d[idx].mean(1); return [round(100 * float(np.quantile(m, q)), 1) for q in (0.025, 0.975)]


def mcnemar_less(a, b):
    """one-sided exact p that arm a has fewer events than b (paired binary)"""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum()); n = n01 + n10
    return sum(comb(n, k) for k in range(0, n01 + 1)) / 2 ** n if n else 1.0, n01, n10


def main():
    data = {k: load(v) for k, v in ARMS.items()}
    eps = [e for e in EPS if all(e in d for d in data.values())]
    col = lambda arm, key: np.array([bool(data[arm][e][key]) for e in eps])
    out = {"protocol": "results/jenga/knock_protocol.json stage2", "paired_episodes": len(eps), "arms": {}}
    for arm in ARMS:
        log = [data[arm][e].get("log", []) for e in eps]
        out["arms"][arm] = {k: int(col(arm, k).sum()) for k in ("success", "red_down", "neighbour_toppled")} | {
            "edits_per_episode": round(float(np.mean([sum(bool(c.get("intervened")) for c in l) for l in log])), 2)}
    for arm in ("ours90", "frag"):
        out[f"{arm}_minus_policy_points"] = {k: {"diff": round(100 * (col(arm, k).mean() - col("policy", k).mean()), 1), "ci95": boot(col(arm, k), col("policy", k))}
                                             for k in ("success", "red_down", "neighbour_toppled")}
    out["frag_minus_ours90_success"] = {"diff": round(100 * (col("frag", "success").mean() - col("ours90", "success").mean()), 1),
                                        "ci95": boot(col("frag", "success"), col("ours90", "success"))}
    p, a, b = mcnemar_less(col("frag", "neighbour_toppled"), col("policy", "neighbour_toppled"))
    out["H_block"] = bool(out["ours90_minus_policy_points"]["red_down"]["diff"] <= -10)
    out["H_frag"] = {"pass": bool(p < 0.05 and out["frag_minus_policy_points"]["red_down"]["diff"] > -10),
                     "mcnemar_p_one_sided": round(p, 5), "frag_only_topples": a, "policy_only_topples": b}
    (ROOT / "results/knock/stage2_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
