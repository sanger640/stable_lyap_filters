"""Benchmark v2 for the optimisation filter redesign (docs/OPT_FILTER_REDESIGN.md, Stage 1).

Decisions: interventions of the SELECTION filter in conf1 (model_OIL, current settings), so the selection baseline is
exactly the chunk it executed in closed loop. Dev set: 150 from episodes 13000-13149; holdout: 150 from 13150-13299
(at most 2 per episode, seeded). Real risk: exact simulator, 32 futures per chunk with paired draws per decision
(same perturbation/noise model as truth_32); two labels: neighbour topple (as truth_32) and red block fallen.

usage: python3 eval/bench2.py prep | truth <dev|holdout> <tag> | baseline <dev|holdout>
"""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/program_b/bench2"
RUN = ROOT / "results/program_b/policy_rollouts/paired_cbfmsoil_conf1"
LOG = ROOT / "results/jenga/paired_cbfmsoil_conf1.jsonl"
N_TRUTH, SEED = 32, 20261701


def _hash(i, chunk):
    return f"{i}:" + hashlib.sha1(np.asarray(chunk, np.float32).round(5).tobytes()).hexdigest()[:16]


def prep():
    import cem_filter as cf
    import inspector_data as insp
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED); rows = []
    for l in LOG.read_text().splitlines():
        r = json.loads(l)
        ks = [c["chunk"] for c in r["log"] if c.get("intervened")]
        for k in (rng.choice(ks, min(2, len(ks)), replace=False).tolist() if ks else []):
            c = next(c for c in r["log"] if c["chunk"] == k)
            rows.append({"ep": r["episode"], "k": int(k), "h_now": c["h_now"], "h_nom": c["h_nom"], "o_nom": c["o_nom"]})
    sets = {}
    for name, lo, hi in (("dev", 13000, 13149), ("holdout", 13150, 13299)):
        R = [x for x in rows if lo <= x["ep"] <= hi]
        sets[name] = [R[j] for j in sorted(rng.choice(len(R), 150, replace=False))]
    mon = cf.Monitor(); sim = insp._sim()
    for name, R in sets.items():
        rec = {k: [] for k in ("state", "ctrl", "target", "gripper", "tilt", "plan16", "nominal", "executed", "site", "z0")}
        for x in R:
            with np.load(RUN / f"ep{x['ep']:03d}.npz") as f:
                S = {k: f[k] for k in f.files}
            d = int(np.flatnonzero(S["chunk_index"] == x["k"])[0]); assert S["intervened"][d]
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
            for k in ("state", "ctrl", "target", "gripper", "plan16", "nominal", "executed"):
                rec[k].append(S[k][d])
            rec["tilt"].append(S["start_tilt"][d]); rec["site"].append(sim.data.site_xpos[sim.site].copy())
            rec["z0"].append(mon.latent(mon.frame(sim)[None])[0].astype(np.float16))
        np.savez_compressed(OUT / f"{name}.npz", **{k: np.stack(v) for k, v in rec.items()})
        (OUT / f"{name}.json").write_text(json.dumps(R))
        print(name, len(R), "decisions", flush=True)


def load(name):
    with np.load(OUT / f"{name}.npz") as f:
        C = {k: f[k] for k in f.files}
    return C, json.loads((OUT / f"{name}.json").read_text())


def hold_chunk(C, i):
    return np.repeat(np.r_[C["target"][i], C["nominal"][i][0, 3]][None], 8, 0).astype(np.float32)


def _truth(job):
    """-> (hash, [neighbour toppled per future], [red block fallen per future])."""
    h, i, state, ctrl, target, gripper, tilt, chunk = job
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import inspector_data as insp
    from fresh_fragility_panel import TOPPLE_DEG, neighbour_tilts
    g = globals()
    if "_S" not in g:
        g["_S"] = insp._sim()
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper)); nb, red = [], []
    for j in range(N_TRUTH):
        tr = um.rollout(g["_S"], snap, chunk, snippets[j] * insp.EXEC_SCALE, [errors.sample(SEED, i, 7, j)])
        nb.append(bool(np.any((neighbour_tilts(tr[-1]) >= TOPPLE_DEG) & (np.asarray(tilt) < TOPPLE_DEG))))
        r = tr[-1][9:15]; red.append(bool(np.degrees(np.arccos(np.clip(r[0] * r[3] - r[2] * r[1], -1, 1))) >= TOPPLE_DEG))
    return h, nb, red


def truth_chunks(name, chunks):
    """chunks: list of (decision index, chunk (8,4)). Cached in OUT/truth_<name>.json: hash -> [p_neighbour, p_red]."""
    C, _ = load(name); f = OUT / f"truth_{name}.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    jobs = []
    for i, ch in chunks:
        h = _hash(i, ch)
        if h not in cache and h not in {j[0] for j in jobs}:
            jobs.append((h, int(i), C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i], np.asarray(ch, np.float32)))
    print(len(jobs), "truth jobs", flush=True)
    if jobs:
        with mp.get_context("spawn").Pool(30) as pool:
            for h, nb, red in pool.imap_unordered(_truth, jobs):
                cache[h] = [float(np.mean(nb)), float(np.mean(red))]
        f.write_text(json.dumps(cache))
    return [cache[_hash(i, ch)] for i, ch in chunks]


def baseline(name):
    """Stage 1: real risk of the policy chunk, the selection filter's executed chunk and holding still."""
    C, R = load(name); n = len(R)
    arms = {"policy": [C["nominal"][i] for i in range(n)], "selection": [C["executed"][i] for i in range(n)],
            "hold": [hold_chunk(C, i) for i in range(n)]}
    res = {}
    for a, chs in arms.items():
        res[a] = np.array(truth_chunks(name, list(enumerate(chs))))
    out = {"set": name, "decisions": n}
    for a, v in res.items():
        out[a] = {"neighbour_risk": round(float(v[:, 0].mean()), 4), "red_fall_risk": round(float(v[:, 1].mean()), 4)}
    d = res["selection"][:, 0] - res["policy"][:, 0]; se = d.std(ddof=1) / np.sqrt(n)
    out["selection_minus_policy_neighbour"] = {"mean": round(float(d.mean()), 4), "one_sided_95_upper": round(float(d.mean() + 1.645 * se), 4),
                                               "better": int((d < 0).sum()), "worse": int((d > 0).sum())}
    (OUT / f"baseline_{name}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prep": prep, "baseline": baseline}[sys.argv[1]](*sys.argv[2:])
