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
    return _SIM


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
        tr = [um.rollout(sim, snap, chunk, snippets[j], [errors.sample(SEED, k, 0, j)]) for j in range(64)]
        af, asafe = labels(events_three_block_v2(np.stack(tr))); kk = topple_count(tr, tilt)
        out[name] = {"any_fork": bool(af), "any_safe": bool(asafe), "topples": int(kk), "harm_fork": bool(2 <= kk <= 62)}
    return k, out


def main(path, workers=int(os.environ.get("LT_WORKERS", 16))):
    path = Path(path)
    with np.load(path, allow_pickle=True) as f:
        S = {k: f[k] for k in f.files}
    jobs = [(k, S["state"][k], S["ctrl"][k], S["target"][k], S["gripper"][k], S["start_tilt"][k], S["nominal"][k], S["executed"][k])
            for k in range(len(S["state"]))]
    res = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (k, out) in enumerate(pool.imap_unordered(job, jobs), 1):
            res[k] = out
            if n % 100 == 0:
                print(f"truth {n}/{len(jobs)}", flush=True)
    keys = ("episode", "chunk_index", "alarm", "intervened", "abstained")
    rows = [{**{kk: (S[kk][k].item() if kk in S else None) for kk in keys}, **res[k]} for k in sorted(res)]
    out = path.with_name(path.stem + "_truth.json")
    out.write_text(json.dumps(rows) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
