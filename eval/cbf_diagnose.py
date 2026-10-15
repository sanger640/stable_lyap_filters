"""Diagnosis of the fork-boundary CBF final test (results/jenga/cbf_final_test_result.json): was each intervention
really safer, why did the filter act, and which "caused" topples come straight from a riskier filter choice.

truth   every one of the 502 interventions: real topple risk (64 futures, the run's 2x execution noise + perception
        error) of the policy's chunk and of the executed chunk, from the saved decision state
report  per-intervention helped / hurt / neutral, by trigger (barrier h vs regime o) and by replacement (policy sample
        vs simple primitive); caused episodes split into direct (a riskier filter choice within 3 decisions before
        the topple) and indirect (runs drifted apart after harmless nudges)
Output: results/jenga/cbf_diagnose.json

usage: EXEC_NOISE_SCALE=2 HARD_MASS=0 python3 eval/cbf_diagnose.py truth|report
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUN = ROOT / "results/program_b/policy_rollouts/paired_cbf_test2x"
TRUTH = ROOT / "results/program_b/cbf_diagnose_truth.json"
OUT = ROOT / "results/jenga/cbf_diagnose.json"
PRIMS = {15: "slow", 16: "vertical", 17: "back off"}
EPS = 0.03                                                    # risk change that counts as real (2 of 64 futures)


def load(tag):
    return {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{tag}.jsonl").read_text().splitlines()}


def truth():
    import inspector_data as insp
    f = load("paired_cbf_test2x")
    done = json.loads(TRUTH.read_text()) if TRUTH.exists() else {}
    jobs = []
    for ep, r in f.items():
        acts = [c["chunk"] for c in r["log"] if c.get("intervened")]
        if not acts:
            continue
        with np.load(RUN / f"ep{ep:03d}.npz") as S:
            for d in acts:
                base = (S["state"][d], S["ctrl"][d], S["target"][d], S["gripper"][d], S["start_tilt"][d])
                for which, ch in ((0, S["nominal"][d]), (1, S["executed"][d])):
                    if f"{ep}:{d}:{which}" not in done:
                        jobs.append(((ep, which, d), *base, ch))
    print(len(jobs), "truth jobs", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("DIAG_WORKERS", 30))) as pool:
        for n, ((ep, which, d), flags) in enumerate(pool.imap_unordered(insp.truth_job, jobs), 1):
            done[f"{ep}:{d}:{which}"] = sum(flags) / 64
            if n % 100 == 0 or n == len(jobs):
                TRUTH.write_text(json.dumps(done)); print(f"truth {n}/{len(jobs)}", flush=True)


def report():
    T = json.loads(TRUTH.read_text())
    a, f = load("paired_record_test2x"), load("paired_cbf_test2x")
    o_max = json.loads((ROOT / "results/jenga/cbf_calibration.json").read_text())["O_MAX"]
    rows = []
    for ep, r in f.items():
        for c in r["log"]:
            if not c.get("intervened"):
                continue
            d = c["chunk"]; rn, rx = T[f"{ep}:{d}:0"], T[f"{ep}:{d}:1"]
            h_trig = c["h_nom"] < min(c["h_now"], 1.0); o_trig = c["o_nom"] > o_max
            rows.append({"ep": ep, "d": d, "risk_policy": rn, "risk_exec": rx,
                         "effect": "helped" if rx < rn - EPS else "hurt" if rx > rn + EPS else "neutral",
                         "trigger": "both" if h_trig and o_trig else "barrier" if h_trig else "regime",
                         "replacement": PRIMS.get(c["choice"], "policy sample"), "kind": c["kind"],
                         "change_mm": round(c["change_mm"], 1), "h_nom": c["h_nom"], "h_choice": c["h_choice"],
                         "o_nom": round(c["o_nom"], 3), "o_choice": round(c["o_choice"], 3)})

    def tab(key):
        out = {}
        for k in sorted({x[key] for x in rows}):
            s = [x for x in rows if x[key] == k]
            out[k] = {"n": len(s), **{e: sum(x["effect"] == e for x in s) for e in ("helped", "hurt", "neutral")},
                      "mean_risk_policy": round(float(np.mean([x["risk_policy"] for x in s])), 3),
                      "mean_risk_exec": round(float(np.mean([x["risk_exec"] for x in s])), 3)}
        return out

    needed = [x for x in rows if x["risk_policy"] >= EPS]
    episodes = {}
    for ep in sorted(set(a) & set(f)):
        ta, tf = a[ep]["neighbour_toppled"], f[ep]["neighbour_toppled"]
        if ta == tf:
            continue
        mine = [x for x in rows if x["ep"] == ep]
        if tf:                                                          # filter arm toppled, policy alone did not
            t0 = next(c["chunk"] for c in f[ep]["log"] if c.get("topple_started"))
            near = [x for x in mine if t0 - 3 <= x["d"] <= t0]
            direct = [x for x in near if x["effect"] == "hurt" or x["risk_exec"] >= 0.25]
            episodes[ep] = {"category": "caused", "topple_chunk": t0, "first_action": min(x["d"] for x in mine),
                            "type": "direct" if direct else "indirect", "culprits": direct}
        else:
            t0 = next(c["chunk"] for c in a[ep]["log"] if c.get("topple_started"))
            near = [x for x in mine if t0 - 3 <= x["d"] <= t0]
            episodes[ep] = {"category": "prevented", "topple_chunk_alone": t0,
                            "acted_within_3_before": bool(near), "helped_near": sum(x["effect"] == "helped" for x in near)}
    caused = [e for e in episodes.values() if e["category"] == "caused"]
    prev = [e for e in episodes.values() if e["category"] == "prevented"]
    res = {"interventions": len(rows),
           "effect": {e: sum(x["effect"] == e for x in rows) for e in ("helped", "hurt", "neutral")},
           "needed (policy's own chunk had >= 3% real risk)": len(needed),
           "effect_when_needed": {e: sum(x["effect"] == e for x in needed) for e in ("helped", "hurt", "neutral")},
           "mean_risk_policy": round(float(np.mean([x["risk_policy"] for x in rows])), 3),
           "mean_risk_exec": round(float(np.mean([x["risk_exec"] for x in rows])), 3),
           "by_trigger": tab("trigger"), "by_replacement": tab("replacement"),
           "caused": {"n": len(caused), "direct": sum(e["type"] == "direct" for e in caused),
                      "indirect": sum(e["type"] == "indirect" for e in caused)},
           "prevented": {"n": len(prev), "acted_within_3_decisions_before_alone_topple": sum(e["acted_within_3_before"] for e in prev)},
           "episodes": episodes, "rows": rows}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("episodes", "rows")}, indent=1))
    for ep, e in episodes.items():
        if e["category"] == "caused":
            print(ep, e["type"], "topple", e["topple_chunk"], "first action", e["first_action"],
                  [(x["d"], x["trigger"], x["replacement"], round(x["risk_policy"], 2), round(x["risk_exec"], 2), x["h_choice"], x["o_choice"]) for x in e["culprits"]])


if __name__ == "__main__":
    {"truth": truth, "report": report}[sys.argv[1]]()
