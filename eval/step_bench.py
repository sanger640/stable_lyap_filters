"""stepbench (results/jenga/stepbench_protocol.json): does a model-step-rate re-check of our fork trigger, with a hold fallback
(the latent safety filter's switching structure, our label-free switch), catch the topples psel's chunk-rate check misses?
From exact saved chunk-start states of paired_cbfps3m_psel: replay the run's commands with its noise; at model steps 2, 4, 6
run the deployed fork_3x trigger on the remaining plan; at the first alarm branch continue vs hold (rest of the chunk), then
the policy alone for 2 chunks, K = 4 policy seeds per branch.
usage: python3 eval/step_bench.py run | analyse -> results/jenga/deep_dive/stepbench.jsonl / results/jenga/stepbench_result.json
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import grip_bench as gb  # noqa: E402

OUT = ROOT / "results/jenga/deep_dive/stepbench.jsonl"
SEED, K, STEPS, TOPPLE = 20261025, 4, (2, 4, 6), 45.0


def cases():
    aud = json.loads((ROOT / "results/jenga/outcome_audit_psel.json").read_text())[gb.RUN]["rows"]
    T, Cpool = [], []
    for r in aud:
        e = r["ep"]
        with np.load(gb.D / f"ep{e}.npz") as S:
            ci = [int(c) for c in S["chunk_index"]]
        if r["logged_neighbour_toppled"]:
            L = json.loads((gb.D / f"ep{e}.json").read_text())["log"]
            kt = next(c["chunk"] for c in L if c.get("topple_started"))
            T += [("T_topple", e, ci.index(kt)), ("T_before", e, ci.index(kt - 1))] if kt - 1 in ci else [("T_topple", e, ci.index(kt))]
        elif r["v4b_success"]:
            Cpool += [("C", e, d) for d in range(1, len(ci))]
    rng = np.random.default_rng(SEED)
    return T + [Cpool[i] for i in rng.choice(len(Cpool), 160, replace=False)]


def trigger(sim, cbf, mon, rem):
    import torch
    z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
    hold = np.repeat(np.r_[sim.target, 0.0][None].astype(np.float32), 16, 0)
    h, o = cbf.score(sim, z0, np.stack([rem, hold]), 64)
    return bool(not cbf.ok(np.array([h[0]]), np.array([o[0]]), h[1])[0]), float(h[0]), float(o[0]), float(h[1])


def run_chunk(sim, cmds, noise, j0, cams, collect=True):
    """execute commands j0..7; -> step of the first neighbour topple (or None), last 2 obs"""
    onset, obs = None, []
    for j in range(j0, 8):
        a = np.asarray(cmds[j], np.float32).copy(); a[:3] -= noise[j]; sim.execute(a)
        if onset is None and float(np.max(sim.block_diagnostics()["tilt"][1:])) >= TOPPLE:
            onset = j
        if collect and j >= 6:
            obs.append(cams.obs())
    return onset, obs


def continuation(sim, cams, pol, hist, noises, k, seed):
    import torch
    import policy_rollout as pr
    torch.manual_seed(seed); hist = list(hist)
    for kk in (k + 1, k + 2):
        chunk = pr.policy_chunk(pol, hist, full=True)[pr.PLAN_START:pr.PLAN_START + 8]
        onset, obs = run_chunk(sim, chunk, noises[kk], 0, cams)
        if onset is not None:
            return True
        hist = (hist + obs)[-2:]
    return False


def run():
    import plausibility as pl
    import fix_counterfactual as fc
    import policy_rollout as pr
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    done = {(r["ep"], r["d"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    mon, cbf, certs = gb.setup(); pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="sb_")))); cams = pr.Cameras(sim)
    for ci_, (kind, e, d) in enumerate(cases()):
        if (e, d) in done:
            continue
        with np.load(gb.D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        k = int(S["chunk_index"][d]); noises = pl.episode_noises(e, gb.LAST + 3, scale=fc.NOISE)
        ex, p16 = np.asarray(S["executed"][d], np.float32), np.asarray(S["plan16"][d], np.float32)
        snap0 = (S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))
        sim.restore(snap0); mon.rng = np.random.default_rng([SEED, e, d])
        checks, jstar, snap_j, onset, obs, t_chk = [], None, None, None, [], []
        for j in range(8):
            if j in STEPS and jstar is None and onset is None:
                rem = np.vstack([np.r_[sim.target, ex[j - 1, 3]][None], ex[j:8], p16[9:16]]).astype(np.float32)
                rem = np.vstack([rem, np.repeat(rem[-1:], 16 - len(rem), 0)])[:16]
                t0 = time.time(); al, hn, on, hh = trigger(sim, cbf, mon, rem); t_chk.append(time.time() - t0)
                checks.append({"j": j, "alarm": al, "h_nom": hn, "o_nom": round(on, 4), "h_now": hh})
                if al:
                    jstar, snap_j = j, sim.snapshot()
            a = ex[j].copy(); a[:3] -= noises[k][j]; sim.execute(a)
            if onset is None and float(np.max(sim.block_diagnostics()["tilt"][1:])) >= TOPPLE:
                onset = j
            if j >= 6:
                obs.append(cams.obs())
        row = {"kind": kind, "ep": e, "d": d, "chunk": k, "onset": onset, "checks": checks, "jstar": jstar,
               "grip_closing": bool(((ex[:, 3] > 0) & (np.r_[ex[0, 3], ex[:-1, 3]] < 0)).any()),
               "sec_check": round(float(np.mean(t_chk)), 2) if t_chk else None}
        if jstar is not None and kind != "C":
            hold = np.array(ex); hold[jstar:, :3] = snap_j[2]; hold[jstar:, 3] = ex[jstar - 1, 3]
            res = {"continue": [], "hold": []}
            for s in range(K):
                for name, cmds in (("continue", ex), ("hold", hold)):
                    sim.restore(snap_j); o_, ob = run_chunk(sim, cmds, noises[k], jstar, cams)
                    hist = ob[-2:]
                    res[name].append(o_ is not None or continuation(sim, cams, pol, hist, noises, k, SEED + 100 * s + 7 * e + d))
            row.update({"topple_continue": res["continue"], "topple_hold": res["hold"]})
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ci_, kind, e, d, "onset", onset, "j*", jstar, row.get("topple_continue"), row.get("topple_hold"), flush=True)
    cams.close(); sim.close(); os._exit(0)


def analyse():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    T = [r for r in rows if r["kind"] != "C"]; C = [r for r in rows if r["kind"] == "C"]
    rng = np.random.default_rng(SEED)
    before = [r for r in T if r["jstar"] is not None and (r["kind"] == "T_before" or r["onset"] is None or r["jstar"] <= r["onset"])]
    eff = np.array([np.mean(r["topple_continue"]) - np.mean(r["topple_hold"]) for r in before if "topple_hold" in r])
    lo = float(np.percentile([eff[rng.integers(len(eff), size=len(eff))].mean() for _ in range(10000)], 5)) if len(eff) else None
    c_alarm = float(np.mean([r["jstar"] is not None for r in C])) if C else None
    out = {"T_cases": len(T), "C_cases": len(C),
           "replay_fidelity_topple_chunk": f'{sum(r["onset"] is not None for r in T if r["kind"] == "T_topple")}/{sum(r["kind"] == "T_topple" for r in T)}',
           "coverage_alarm_before_onset": len(before),
           "P_topple_continue": round(float(np.mean([np.mean(r["topple_continue"]) for r in before])), 3) if before else None,
           "P_topple_hold": round(float(np.mean([np.mean(r["topple_hold"]) for r in before])), 3) if before else None,
           "effect_lower95_one_sided": round(lo, 3) if lo is not None else None,
           "control_step_alarm_share": round(c_alarm, 3) if c_alarm is not None else None,
           "alarms_by_step": {str(j): {t: sum(r["jstar"] == j for r in rows if (r["kind"] == "C") == (t == "C")) for t in ("T", "C")} for j in STEPS},
           "control_alarms_with_grip_closing": sum(r["jstar"] is not None and r["grip_closing"] for r in C),
           "sec_check_median": float(np.median([r["sec_check"] for r in rows if r["sec_check"]]))}
    out["PRIMARY_pass"] = bool(len(before) >= 8 and lo is not None and lo > 0 and c_alarm is not None and c_alarm <= 0.15)
    (ROOT / "results/jenga/stepbench_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1:] == ["run"] else analyse()
