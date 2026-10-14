"""Why does filter v2 not reduce topples in closed loop? Uses per-decision ground truth (eval/loop_truth.py) of the paired
runs: policy alone (paired_monitor, alarms logged) and policy + CEM v2 (paired_cem_v2).

Q1  Does the monitor flag the decisions that matter? Recall/precision of alarms against true forks, and for the chunk in
    which each real topple started: was it flagged, and was it a fork or a near-certain topple (>= 62/64)?
Q2  When the filter changed an action, was the replacement truly safer? True topple probability nominal vs executed.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/program_b/policy_rollouts"


def load(arm):
    rows = json.loads((R / f"{arm}_truth.json").read_text())
    eps = {json.loads(l)["episode"]: json.loads(l) for l in (ROOT / f"results/jenga/{arm}.jsonl").read_text().splitlines()}
    return rows, eps


def main():
    out = {}
    rows, eps = load("paired_monitor")
    p = lambda r, k="nominal": r[k]["topples"] / 64
    fork = np.array([r["nominal"]["harm_fork"] for r in rows]); alarm = np.array([bool(r["alarm"]) for r in rows])
    risky = np.array([p(r) >= 0.03 for r in rows])
    out["Q1_monitor_vs_truth"] = {
        "decisions": len(rows), "true_topple_forks": int(fork.sum()), "decisions_with_any_topple_risk": int(risky.sum()),
        "alarms": int(alarm.sum()), "recall_on_topple_forks": round(float((alarm & fork).sum() / max(fork.sum(), 1)), 3),
        "precision_vs_topple_forks": round(float((alarm & fork).sum() / max(alarm.sum(), 1)), 3),
        "alarm_rate_on_no_risk": round(float((alarm & ~risky).sum() / max((~risky).sum(), 1)), 3)}
    starts = []
    for e, ep in eps.items():
        tc = [c["chunk"] for c in ep["log"] if c.get("topple_started")]
        for c in tc:
            r = next((r for r in rows if r["episode"] == e and r["chunk_index"] == c), None)
            if r:
                starts.append({"episode": e, "chunk": c, "flagged": bool(r["alarm"]), "true_topple_prob": round(p(r), 3)})
    out["Q1_chunks_where_topples_started"] = {
        "count": len(starts), "flagged": sum(s["flagged"] for s in starts),
        "near_certain_topple_(>=0.97)": sum(s["true_topple_prob"] >= 0.97 for s in starts),
        "fork_(0.03-0.97)": sum(0.03 <= s["true_topple_prob"] < 0.97 for s in starts),
        "looked_safe_(<0.03)": sum(s["true_topple_prob"] < 0.03 for s in starts), "rows": starts}
    rows2, eps2 = load("paired_cem_v2")
    ch = [r for r in rows2 if r.get("intervened") and not r.get("abstained")]
    ab = [r for r in rows2 if r.get("abstained")]
    a = np.array([p(r) for r in ch]); b = np.array([p(r, "executed") for r in ch])
    out["Q2_filter_replacements"] = {
        "changed_actions": len(ch), "holds": len(ab),
        "true_topple_prob_original_mean": round(float(a.mean()), 4) if len(a) else None,
        "true_topple_prob_replacement_mean": round(float(b.mean()), 4) if len(b) else None,
        "replacement_safer": int((b < a - 0.03).sum()), "replacement_riskier": int((b > a + 0.03).sum()),
        "about_the_same": int((np.abs(b - a) <= 0.03).sum()),
        "original_had_no_risk": int((a < 0.03).sum()),
        "holds_true_topple_prob_original_mean": round(float(np.mean([p(r) for r in ab])), 4) if ab else None}
    (ROOT / "results/jenga/loop_diagnose.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "rows"}) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
