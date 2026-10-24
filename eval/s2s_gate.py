"""Primary analysis of results/jenga/s2s_protocol.json (written before any shifted-physics episode was seen).
fork_3x rule per model (its own label-free O_MAX) on the evaluation episodes' margin_check rows: safe alarm rate and share of
topple episodes warned >= 3 chunks ahead; model_S2S minus model_OIL paired over topple episodes (bootstrap lower bound).
Also the stage-0 reference (normal physics, paired_record_scl) when its rows exist.
usage: python3 eval/s2s_gate.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import seed_pool_check as sp  # noqa: E402

D = ROOT / "results/jenga/deep_dive"
RUN = ROOT / "results/program_b/policy_rollouts"


def rows(folder):
    return {(r["ep"], r["k"]): r for f in sorted(folder.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}


def evaluate(M, run, audit, om):
    A = json.loads(audit.read_text())[run]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    keys = sorted(k for k in M if k[0] in top)
    kt = {e: next((c["chunk"] for c in json.loads((RUN / run / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None)
          for e in {k[0] for k in keys}}
    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys])
    o = np.array([M[k]["o"] for k in keys]); fire = (hp < np.minimum(hh, 2.0) - 1e-9) | (o > om)
    safe = np.array([not top[k[0]] for k in keys])
    by = {}
    for (e, k), f in zip(keys, fire):
        by.setdefault(e, {})[k] = bool(f)
    per = {}
    for e, ks in by.items():
        if top[e] and kt.get(e) is not None:
            h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
            per[e] = bool(h) and kt[e] - h[0] >= 3
    return {"decisions": len(keys), "safe_alarm_rate": round(float(fire[safe].mean()), 4), "topple_episodes": len(per),
            "lead_ge3_share": round(float(np.mean(list(per.values()))), 3)}, per


def main():
    run, audit = "paired_record_s2s", ROOT / "results/jenga/outcome_audit_s2s.json"
    om_oil = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]
    om_s2s = json.loads((ROOT / "results/jenga/cbf_calibration_S2S.json").read_text())["O_MAX"]
    base, pb = evaluate(rows(D / "s2s"), run, audit, om_oil)
    adap, pa = evaluate(rows(D / "s2s_adapt"), run, audit, om_s2s)
    E = sorted(set(pb) & set(pa)); x = np.array([pa[e] for e in E], float) - np.array([pb[e] for e in E], float)
    rng = np.random.default_rng(20261124); bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
    out = {"model_OIL": base, "model_S2S": adap, "O_MAX": {"OIL": om_oil, "S2S": om_s2s},
           "S2S_minus_OIL_lead_pp": round(100 * float(x.mean()), 1), "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 1),
           "s2s_only": int((x > 0).sum()), "oil_only": int((x < 0).sum())}
    out["PRIMARY_pass"] = bool(out["lower95_one_sided_pp"] > 0 and adap["safe_alarm_rate"] <= base["safe_alarm_rate"] + 0.02)
    ref = D / "scl"
    if ref.exists():
        out["stage0_normal_physics_reference_OIL"], _ = evaluate(rows(ref), "paired_record_scl", ROOT / "results/jenga/outcome_audit_scl.json", om_oil)
    err = D / "s2s_error.json"
    if err.exists():
        out["latent_error"] = json.loads(err.read_text())
    (ROOT / "results/jenga/s2s_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
