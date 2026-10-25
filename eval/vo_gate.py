"""Gate of results/jenga/vo_protocol.json: exact-sim oracle sensitivity vs the world model's relative total variance as
an early-warning alarm on the held-out scl decisions.  usage: python3 eval/vo_gate.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/jenga/deep_dive"; RUN = ROOT / "results/program_b/policy_rollouts/paired_record_scl"


def main(sub="vo_scl"):
    V = {(r["ep"], r["k"]): r for f in sorted((D / sub).glob("rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    W = {(r["ep"], r["k"]): r for r in map(json.loads, (D / "spread_rows_scl.jsonl").read_text().splitlines())}
    keys = sorted(set(V) & set(W))
    A = json.loads((ROOT / "results/jenga/outcome_audit_scl.json").read_text())["paired_record_scl"]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None) for e in {k[0] for k in keys}}
    safe = np.array([not top[k[0]] for k in keys])
    y = np.array([V[k]["y"] for k in keys])
    tv = np.array([np.log(W[k]["stats"]["4.0"]["tvar30"][0] + 1e-12) - np.log(W[k]["stats"]["4.0"]["tvar30"][1] + 1e-12) for k in keys])

    def ev(f):
        by = {}
        for (e, k), x in zip(keys, f):
            by.setdefault(e, {})[k] = bool(x)
        per = {}
        for e, ks in by.items():
            if top[e] and kt.get(e) is not None:
                h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                per[e] = bool(h) and kt[e] - h[0] >= 3
        return float(np.mean(np.asarray(f)[safe])), per
    from scipy.stats import spearmanr
    out = {"decisions": len(keys), "of": len(W), "spearman_y_tvrel": round(float(spearmanr(y, tv).correlation), 3),
           "share_y_zero": round(float(np.mean(np.abs(y) < 1e-9)), 3)}
    rng = np.random.default_rng(20261030)
    for name, rate in (("fork_2x", 0.062), ("fork_3x", 0.1133), ("fork_4x", 0.150)):
        ro, po = ev(y > np.quantile(y[safe], 1 - rate)); rw, pw = ev(tv > np.quantile(tv[safe], 1 - rate))
        E = sorted(po); x = np.array([po[e] for e in E], float) - np.array([pw[e] for e in E], float)
        bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
        out[name + "_rate"] = {"target_rate": rate, "oracle_rate": round(ro, 4), "wm_rate": round(rw, 4), "topple_episodes": len(E),
                               "oracle_lead_ge3": round(float(np.mean([po[e] for e in E])), 3), "wm_tvrel_lead_ge3": round(float(np.mean([pw[e] for e in E])), 3),
                               "oracle_minus_wm_pp": round(100 * float(x.mean()), 1), "lower95_pp": round(100 * float(np.percentile(bs, 5)), 1)}
    out["PRIMARY_pass"] = bool(out["fork_3x_rate"]["lower95_pp"] > 0)
    (ROOT / f"results/jenga/{'vo' if sub == 'vo_scl' else sub}_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
