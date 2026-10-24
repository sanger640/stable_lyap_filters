"""Counterfactual cost of one filter fix on the approach (results/jenga/fixcf_protocol.json).
From saved decision states of the 3x filter's stl episodes (paired_cbfg3m_stl, no neighbour topple), at a decision where the
filter intervened before the first grasp: rebuild the policy's exact 2-frame observation history (replay decision d-1),
then from the exact saved state d execute ONE chunk three ways - the filter's fix, the policy's own nominal chunk, a hold - with the run's execution
noise, and let the POLICY ALONE continue (same denoising seed and noise in every branch) for up to M chunks.
Per branch: chunks until the first gripper close, extraction, neighbour topple.

usage: python3 eval/fix_counterfactual.py run    (SHARD=i/n)   -> results/jenga/deep_dive/fixcf/rows_<i>.jsonl
       python3 eval/fix_counterfactual.py cases                -> prints the case count
"""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUN = "paired_cbfg3m_stl"
D = ROOT / "results/program_b/policy_rollouts" / RUN
OUT = ROOT / "results/jenga/deep_dive/fixcf"
N_CASES, M, SEED = 150, 30, 20261025
NOISE = float(os.environ.get("EXEC_NOISE_SCALE", 2.0))


def cases():
    A = {r["ep"]: r for r in json.loads((ROOT / "results/jenga/outcome_audit_stl.json").read_text())[RUN]["rows"]}
    rng = np.random.default_rng(SEED); out = []
    for e in sorted(A):
        if A[e]["logged_neighbour_toppled"]:
            continue
        with np.load(D / f"ep{e}.npz") as f:
            g = f["gripper"] < 55; I = f["intervened"]
        k = int(np.argmax(g)) if g.any() else len(g)
        ds = [d for d in range(1, k) if I[d]]
        if ds:
            out.append((e, int(rng.choice(ds))))
    idx = rng.permutation(len(out))[:N_CASES]
    return [out[i] for i in sorted(idx)]


def run():
    import plausibility as pl
    import policy_rollout as pr
    import torch
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    shard, nshard = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    OUT.mkdir(parents=True, exist_ok=True); fout = OUT / f"rows_{shard}.jsonl"
    done = {(r["ep"], r["d"]) for r in map(json.loads, fout.read_text().splitlines())} if fout.exists() else set()
    pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="fcf_")))); cams = pr.Cameras(sim)
    logs = {}
    for ci, (e, d) in enumerate(cases()):
        if ci % nshard != shard or (e, d) in done:
            continue
        with np.load(D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        logs = json.loads((D / f"ep{e}.json").read_text())["log"]
        noises = pl.episode_noises(e, int(S["chunk_index"][d]) + M + 2, scale=NOISE)
        row = {"ep": e, "d": d, "chunk": int(S["chunk_index"][d]), "progress": logs[int(S["chunk_index"][d])].get("progress"),
               "fix_kind": logs[int(S["chunk_index"][d])].get("fix_kind")}
        sim.restore((S["state"][0], S["ctrl"][0], S["target"][0], float(S["gripper"][0])))
        xy0 = sim.data.xpos[sim.tracked_block_ids[0]][:2].copy()
        hold = np.repeat(np.r_[S["target"][d], 0.0][None], 8, 0).astype(np.float32)
        for name, first in (("fix", S["executed"][d]), ("nominal", S["nominal"][d]), ("hold", hold)):
            sim.restore((S["state"][d - 1], S["ctrl"][d - 1], S["target"][d - 1], float(S["gripper"][d - 1])))
            hist = []
            for j, a in enumerate(S["executed"][d - 1]):                     # exact history: replay chunk d-1
                a = np.asarray(a, np.float32).copy(); a[:3] -= noises[int(S["chunk_index"][d - 1])][j]; sim.execute(a)
                if j >= 6:
                    hist.append(cams.obs())
            hand = sim.data.site_xpos[sim.site].copy()
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))   # branch from the exact saved state
            drift = round(1000 * float(np.abs(hand - sim.data.site_xpos[sim.site]).max()), 3) if name == "fix" else None   # history frames' hand error, mm
            z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
            res = {"grasp": None, "extract": None, "topple": None}
            torch.manual_seed(int(SEED + 1000 * e + d))
            for c in range(M + 1):
                k = int(S["chunk_index"][d]) + c
                chunk = first if c == 0 else pr.policy_chunk(pol, hist, full=True)[pr.PLAN_START:pr.PLAN_START + 8]
                for j, a in enumerate(chunk):
                    a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a); hist.append(cams.obs())
                    if res["grasp"] is None and sim.gripper < 55:
                        res["grasp"] = c
                    p = sim.data.xpos[sim.tracked_block_ids[0]]
                    if res["extract"] is None and (p[2] - z0 >= 0.02 or p[0] - xy0[0] <= -0.02):
                        res["extract"] = c
                    if res["topple"] is None and float(np.max(sim.block_diagnostics()["tilt"][1:])) >= 45:
                        res["topple"] = c
                hist = hist[-2:]
                if res["extract"] is not None or res["topple"] is not None:
                    break
            row[name] = res
            if drift is not None:
                row["history_replay_drift_mm"] = drift
        with open(fout, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ci, row, flush=True)
    cams.close(); sim.close()


def analyse():
    R = [r for f in sorted(OUT.glob("rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())]
    g = lambda b: b["grasp"] if b["grasp"] is not None else M + 1
    ok = [r for r in R if r["fix"]["topple"] is None and r["nominal"]["topple"] is None]
    dfix = np.array([g(r["fix"]) - g(r["nominal"]) for r in ok], float)
    lost = np.array([1 - min(1.0, max(0.0, r["progress"] if r["progress"] is not None else 0.0)) for r in ok])
    ex = dfix - lost
    rng = np.random.default_rng(20261026); bs = np.array([ex[rng.integers(len(ex), size=len(ex))].mean() for _ in range(10000)])
    hold_ok = [r for r in ok if r["hold"]["topple"] is None]
    out = {"cases": len(R), "no_topple_either": len(ok), "mean_delay_fix": round(float(dfix.mean()), 3), "median_delay_fix": float(np.median(dfix)),
           "mean_lost_motion": round(float(lost.mean()), 3), "mean_excess": round(float(ex.mean()), 3),
           "excess_lower95": round(float(np.percentile(bs, 5)), 3), "excess_upper95": round(float(np.percentile(bs, 95)), 3),
           "mean_delay_hold": round(float(np.mean([g(r["hold"]) - g(r["nominal"]) for r in hold_ok])), 3),
           "censored": {b: sum(r[b]["grasp"] is None for r in R) for b in ("fix", "nominal", "hold")},
           "topple": {b: sum(r[b]["topple"] is not None for r in R) for b in ("fix", "nominal", "hold")},
           "nominal_topples_fix_does_not": sum(r["nominal"]["topple"] is not None and r["fix"]["topple"] is None for r in R),
           "extract_within_30": {b: sum(r[b]["extract"] is not None for r in R) for b in ("fix", "nominal", "hold")},
           "delay_fix_hist": {str(int(v)): int((dfix == v).sum()) for v in np.unique(dfix)},
           "max_history_replay_drift_mm": max(r.get("history_replay_drift_mm", 0) for r in R)}
    out["verdict"] = "amplification" if out["excess_lower95"] > 0 else "no amplification" if out["excess_upper95"] < 0.5 else "inconclusive"
    (ROOT / "results/jenga/fixcf_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "cases":
        C = cases(); print(len(C), C[:5])
    else:
        {"run": run, "analyse": analyse}[sys.argv[1]]()
