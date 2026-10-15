"""Per-decision ground truth for closed-loop runs: for every saved decision state (policy_rollout.episode(snaps=...)),
64 MuJoCo futures from the true state with the corrected perception error and the execution-noise snippets, for the
policy's nominal chunk and for the chunk actually executed. Labels: gripper-relative any-outcome fork (primary) and
neighbour-topple fork (harm). CPU pool.

usage: python eval/loop_truth.py <snaps.npz> [workers]   -> <snaps>_truth.json
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
SEED = 20261520
_SIM = None


def _sim():
    global _SIM
    if _SIM is None:
        import uncertainty_monitor_dev as um
        from jenga_short_held_tails import DirectJengaSim, extract_sim
        _SIM = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="lt_"))))
        ms = float(os.environ.get("HARD_MASS", 0))           # same physics as the run (stress layouts: light neighbours)
        if ms:
            m = _SIM.model
            for name in ("block_left", "block_right"):
                b = m.body(name).id
                m.body_mass[b] *= ms; m.body_inertia[b] *= ms
    return _SIM


def fast_rollout(sim, snap, chunk, noise, err):
    """As uncertainty_monitor_dev.rollout (chunk with execution noise, then hold the last target), but the hold stops
    early once every block has been still for 5 steps (LT_EARLY_STOP=1); the remaining steps repeat the last state."""
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    from jenga_state_data import step_state
    if os.environ.get("LT_EARLY_STOP", "1") != "1":                   # default on: identical results, ~1.7x faster
        return um.rollout(sim, snap, chunk, noise, [err])
    sim.restore(snap); pr.perturb_blocks(sim, *err)
    a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= noise
    out = []
    for x in a:
        sim.execute(x); out.append(step_state(sim))
    still = 0
    for _ in range(um.HOLD):
        if still >= 5:
            out.append(out[-1]); continue
        sim.execute(chunk[-1]); out.append(step_state(sim))
        v = np.abs(out[-1][27:45]).max()                   # block linear + angular velocities
        still = still + 1 if v < 1e-3 else 0
    return np.stack(out)


def job(args):
    k, state, ctrl, target, gripper, tilt, nominal, executed = args
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from regrade_any_outcome import events_three_block_v2, labels
    assert pr.VALID_PERTURBATION
    sim = _sim(); errors = p3.ErrorModel(); snippets = np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    out = {}
    for name, chunk in (("nominal", nominal), ("executed", executed)):
        if name == "executed" and np.allclose(executed, nominal):
            out["executed"] = out["nominal"]; continue
        nf = int(os.environ.get("LT_FUTURES", 64))          # screening: 32; reported results: 64
        tr = [fast_rollout(sim, snap, chunk, snippets[j], errors.sample(SEED, int(k), 0, j)) for j in range(nf)]
        af, asafe = labels(events_three_block_v2(np.stack(tr))); kk = round(topple_count(tr, tilt) * 64 / nf)
        out[name] = {"any_fork": bool(af), "any_safe": bool(asafe), "topples": int(kk), "harm_fork": bool(2 <= kk <= 62)}
    return k, out


def main(path, workers=int(os.environ.get("LT_WORKERS", 16))):
    path = Path(path)
    if path.is_dir():                                        # a paired-run folder of per-episode files
        parts = [dict(np.load(p_, allow_pickle=True)) for p_ in sorted(path.glob("ep*.npz"))]
        S = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        path = path.with_suffix(".npz")
    else:
        with np.load(path, allow_pickle=True) as f:
            S = {k: f[k] for k in f.files}
    keep = np.arange(len(S["state"]))
    if os.environ.get("LT_SUBSET"):                          # decisions that matter + a random sample of the rest
        import json as _j
        arm = path.stem
        eps = {(_r := _j.loads(l))["episode"]: _r for l in (ROOT / f"results/jenga/{arm}.jsonl").read_text().splitlines()}
        started = {(e, c["chunk"]) for e, r in eps.items() for c in r["log"] if c.get("topple_started")}
        pre = {(e, c - 1) for e, c in started} | {(e, c - 2) for e, c in started}
        imp = np.array([bool(S["alarm"][k] or S["intervened"][k] or (int(S["episode"][k]), int(S["chunk_index"][k])) in started | pre)
                        for k in range(len(S["state"]))])
        rest = np.flatnonzero(~imp)
        samp = np.random.default_rng(20261540).choice(rest, min(len(rest), int(os.environ.get("LT_SAMPLE", 500))), replace=False)
        keep = np.sort(np.concatenate([np.flatnonzero(imp), samp]))
        S["sampled_background"] = np.isin(np.arange(len(S["state"])), samp)
        print("subset", len(keep), "of", len(S["state"]), "decisions", flush=True)
    jobs = [(k, S["state"][k], S["ctrl"][k], S["target"][k], S["gripper"][k], S["start_tilt"][k], S["nominal"][k], S["executed"][k])
            for k in keep]
    res = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (k, out) in enumerate(pool.imap_unordered(job, jobs), 1):
            res[k] = out
            if n % 100 == 0:
                print(f"truth {n}/{len(jobs)}", flush=True)
    keys = ("episode", "chunk_index", "alarm", "intervened", "abstained", "sampled_background")
    rows = [{**{kk: (S[kk][k].item() if kk in S else None) for kk in keys}, **res[k]} for k in sorted(res)]
    out = path.with_name(path.stem + "_truth.json")
    out.write_text(json.dumps(rows) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
