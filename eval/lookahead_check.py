"""Step 2 (overnight plan 2026-10-15): does a longer look-ahead give the fork-boundary barrier an earlier warning?

Closed-loop recordings at normal mass (paired_record_normal: policy alone, every decision state and the policy's full
16-step plan saved). For decisions in the 3 chunks before each real neighbour topple, plus a random sample of all
decisions:
  truth   64 MuJoCo futures (corrected perception error + execution noise, normal physics) for two horizons:
          H8  = the executed chunk (plan steps 1..8), then hold 30            (what the monitor imagines today)
          H15 = the policy's whole remaining plan (steps 1..15), then hold 23  (same total length, 38 commands)
  latent  the barrier (distance to the fork in units of uncertainty) and the latent change outside the robot's hand
          path, for both horizons, purely from the world model.
Labels are used only to evaluate.

stages:  truth (CPU pool) | latent (GPU) | report
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
TAG = os.environ.get("LA_TAG", "paired_record_dev2x")           # recorded run (policy alone)
RUN = ROOT / "results/program_b/policy_rollouts" / TAG
OUT = ROOT / f"results/jenga/lookahead_check_{TAG}.json"
EXEC_SCALE = float(os.environ.get("LA_EXEC_SCALE", 2.0))        # ground truth uses the run's execution-noise level
SEED = 20261580
PLAN_START = 1


def horizon_actions(plan, horizon, noise=None):
    """(38, 4) commands: H8 = plan[1:9] + 30 x hold, H15 = plan[1:16] + 23 x hold; noise (k, 3) subtracted from the moving part."""
    n = 8 if horizon == "H8" else 15
    a = np.asarray(plan[PLAN_START:PLAN_START + n], np.float32).copy()
    if noise is not None:
        a[:, :3] -= noise[:n]
    hold = np.repeat(np.asarray(plan[PLAN_START + n - 1], np.float32)[None], 38 - n, 0)
    return np.concatenate([a, hold])


def noise_for(snippets, j):
    return np.concatenate([snippets[j], snippets[(j + 1) % len(snippets)]])     # 16 x 3


def decisions(sample=300):
    eps = {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{TAG}.jsonl").read_text().splitlines()}
    starts = {(e, c["chunk"]) for e, r in eps.items() for c in r["log"] if c.get("topple_started")}
    want = set()
    for e, c in starts:
        want |= {(e, c - d) for d in (0, 1, 2, 3)}
    allk = []
    for p in sorted(RUN.glob("ep*.npz")):
        e = int(p.stem[2:])
        with np.load(p) as f:
            allk += [(e, int(c)) for c in f["chunk_index"]]
    rng = np.random.default_rng(SEED)
    rest = [k for k in allk if k not in want]
    bg = [rest[i] for i in rng.choice(len(rest), min(sample, len(rest)), replace=False)]
    keys = sorted({k for k in want if k in set(allk)} | set(bg))
    return keys, starts, set(bg), eps


_S = None


def truth_job(args):
    key, state, ctrl, target, gripper, tilt, plan = args
    global _S
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import step_state
    from regrade_any_outcome import events_three_block_v2, labels
    if _S is None:
        _S = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="la_"))))
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    out = {}
    for hz in ("H8", "H15"):
        trs = []
        for j in range(64):
            _S.restore(snap); pr.perturb_blocks(_S, *errors.sample(SEED, key[0], key[1], j))
            cmds = horizon_actions(plan, hz, noise_for(snippets, j) * EXEC_SCALE)
            tr, still = [], 0
            for t, a in enumerate(cmds):
                if t >= 16 and still >= 5:
                    tr.append(tr[-1]); continue
                _S.execute(a); tr.append(step_state(_S))
                still = still + 1 if np.abs(tr[-1][27:45]).max() < 1e-3 else 0
            trs.append(np.stack(tr))
        k = topple_count(trs, tilt); af, asafe = labels(events_three_block_v2(np.stack(trs)))
        out[hz] = {"topples": int(k), "any_fork": bool(af), "any_safe": bool(asafe)}
    return key, out


def load_state(key, cache={}):
    e, c = key
    if e not in cache:
        cache.clear(); cache[e] = dict(np.load(RUN / f"ep{e:03d}.npz"))
    S = cache[e]; i = int(np.flatnonzero(S["chunk_index"] == c)[0])
    return S, i


def stage_truth(workers=int(os.environ.get("LA_WORKERS", 28))):
    keys, starts, bg, _ = decisions()
    jobs = []
    for k in keys:
        S, i = load_state(k)
        jobs.append((k, S["state"][i], S["ctrl"][i], S["target"][i], S["gripper"][i], S["start_tilt"][i], S["plan16"][i]))
    res = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (k, o) in enumerate(pool.imap_unordered(truth_job, jobs), 1):
            res[f"{k[0]}:{k[1]}"] = o
            if n % 50 == 0:
                print(f"truth {n}/{len(jobs)}", flush=True)
    data = {"keys": [list(k) for k in keys], "starts": [list(k) for k in starts], "background": [list(k) for k in bg], "truth": res}
    OUT.write_text(json.dumps(data) + "\n")
    print("truth done", len(res), flush=True)


def stage_latent():
    import cem_filter as cf
    import cbf_barrier as cb
    import step3_world_model as wm
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    import uncertainty_monitor_dev as um
    data = json.loads(OUT.read_text())
    mon = cf.Monitor(seed=SEED + 1); torch = mon.torch
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="lal_"))))
    lat = {}
    for n, k in enumerate(map(tuple, data["keys"])):
        S, i = load_state(k)
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        frame = mon.frame(sim)
        z0 = torch.as_tensor(mon.latent(frame[None])[0], device="cuda").float()
        plan = S["plan16"][i]
        res = {}
        for hz in ("H8", "H15"):
            split, outside = [], None
            for s in cb.SCALES:
                d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                               for p, r in (mon.errors.sample(SEED + 2, k[0], k[1], j) for j in range(64))])
                st = mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
                acts = np.stack([horizon_actions(plan, hz, noise_for(mon.snippets, j) * s) for j in range(64)])
                pairs = torch.as_tensor(acts, device="cuda").reshape(64, wm.STEPS, -1)
                z = st.float(); ends = {}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in range(wm.STEPS):
                        z = mon.model.step(z, pairs[:, t]).float()
                        if t in (wm.HOLD10, wm.HOLD30):
                            ends[t] = z.reshape(1, 64, -1)
                ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
                split.append(bool((ok10 & ok30).item()))
                if s == 1.0:
                    pts = np.vstack([sim.data.site_xpos[sim.site][None], plan[PLAN_START:PLAN_START + (8 if hz == "H8" else 15), :3]])
                    m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                    diff = (ends[wm.HOLD30][0].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                    outside = ((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).cpu().numpy()
            hit = [i_ for i_, v in enumerate(split) if v]
            res[hz] = {"h": float(np.log2(cb.SCALES[hit[0]])) if hit else float(np.log2(cb.SCALES[-1])), "split": split,
                       "outside_max": float(outside.max()), "outside_q90": float(np.quantile(outside, 0.9))}
        lat[f"{k[0]}:{k[1]}"] = res
        torch.cuda.empty_cache()
        if n % 50 == 0:
            print(f"latent {n}/{len(data['keys'])}", flush=True)
    data["latent"] = lat
    OUT.write_text(json.dumps(data) + "\n")
    print("latent done", flush=True)


def stage_report():
    from sklearn.metrics import roc_auc_score
    data = json.loads(OUT.read_text())
    T, L = data["truth"], data["latent"]
    keys = [k for k in T if k in L]
    rep = {"decisions": len(keys), "topple_starts": len(data["starts"])}
    for hz in ("H8", "H15"):
        risk = np.array([T[k][hz]["topples"] / 64 for k in keys]); y = risk >= 0.03
        r = {"risky": int(y.sum())}
        if 0 < y.sum() < len(y):
            for sig, f in (("barrier", lambda k: -L[k][hz]["h"]), ("split_1x", lambda k: L[k][hz]["split"][1]),
                           ("outside_max", lambda k: L[k][hz]["outside_max"])):
                r["auc_" + sig] = round(roc_auc_score(y, [f(k) for k in keys]), 3)
        rep[hz] = r
    early = []
    for e, c in data["starts"]:
        row = {"episode": e, "chunk": c}
        for d in (3, 2, 1, 0):
            k = f"{e}:{c - d}"
            if k in T and k in L:
                row[f"c-{d}"] = {hz: {"risk": round(T[k][hz]["topples"] / 64, 2), "h": L[k][hz]["h"],
                                      "out": round(L[k][hz]["outside_max"], 3)} for hz in ("H8", "H15")}
        early.append(row)
    rep["before_topples"] = early
    (ROOT / f"results/jenga/lookahead_report_{TAG}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "before_topples"}, indent=1), flush=True)
    for row in early:
        print(row["episode"], row["chunk"], " | ".join(f"{d}: H8 r{row[d]['H8']['risk']} h{row[d]['H8']['h']:.1f}  H15 r{row[d]['H15']['risk']} h{row[d]['H15']['h']:.1f}"
                                                       for d in ("c-3", "c-2", "c-1", "c-0") if d in row), flush=True)


if __name__ == "__main__":
    {"truth": stage_truth, "latent": stage_latent, "report": stage_report}[sys.argv[1]]()
