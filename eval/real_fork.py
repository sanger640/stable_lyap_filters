"""Deep dive, decomposition (docs/FILTER_DEEP_DIVE.md): the certificate's criterion on REAL futures. Same perturbations as
cbf.score (planar pose error + yaw from mon.errors, execution-noise snippets on the 15 look-ahead actions, scales 0.5 / 1 / 2,
n futures), but the futures come from the exact simulator, are rendered with the monitor's close camera at the same two
times (18 and 38 actions) and encoded: (a) PCA8 latent (as the world model sees), (b) full DINOv2 ViT-S tokens.
h, o computed exactly as cbf.score (split test at both times; o = max over futures of mean outside-hand change at 38).
usage: python3 eval/real_fork.py <set> <n_decisions> [n_fut]
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
SCALES = (0.5, 1.0, 2.0)
OUT = ROOT / "results/jenga/deep_dive"
_W = {}


def _job(args):
    """(key, snap, actions (38, 4), perturbation) -> (key, frame at 18, frame at 38)."""
    key, snap, acts, pert = args
    if not _W:
        import cem_filter as cf
        import inspector_data as insp
        _W["sim"] = insp._sim(); _W["mon"] = cf.Monitor.__new__(cf.Monitor); _W["mon"].renderer = None
    import program_b_reference as pr
    import cem_filter as cf
    sim, mon = _W["sim"], _W["mon"]
    sim.restore(snap); pr.perturb_blocks(sim, *pert); frames = []
    for t, a in enumerate(acts):
        sim.execute(a)
        if t + 1 in (18, 38):
            frames.append(cf.Monitor.frame(mon, sim))
    return key, frames[0], frames[1]


def main(name, n_dec, n_fut=32):
    import torch
    import bench2 as b2
    import cbf_barrier as cb
    import cem_filter as cf
    import cbf_filter as cbff
    import rank_check as rc
    from jenga_vjepa_encode import dino_tokens
    n_dec, n_fut = int(n_dec), int(n_fut)
    C, cand = rc.chunks_for(name); mon = cf.Monitor(); L = mon.latent
    dec = [i for i in sorted(cand) if len(cand[i]) >= 2]
    if os.environ.get("REAL_INFORMATIVE") == "1":            # only decisions whose chunks differ in real risk (>= 2 futures)
        dec = [i for i in dec if max(r for _, r in cand[i].values()) - min(r for _, r in cand[i].values()) >= 2 / 32 - 1e-9]
    dec = sorted(np.random.default_rng(20261041).choice(dec, min(n_dec, len(dec)), replace=False).tolist())
    f = OUT / f"real_{name}_rows.jsonl"; done = {json.loads(l)["i"] for l in f.read_text().splitlines()} if f.exists() else set()
    pool = mp.get_context("spawn").Pool(24)
    for i in dec:
        if i in done:
            continue
        snap = (C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i]))
        import inspector_data as insp
        sim = _W.setdefault("main_sim", insp._sim()); sim.restore(snap)
        site = sim.data.site_xpos[sim.site].copy(); f0 = mon.frame(sim)
        z0 = torch.as_tensor(L(f0[None])[0], device="cuda").float()
        t0 = dino_tokens(L.encoder, f0[None], "cuda")[0].float()                     # (784, 384)
        names = list(cand[i]); tgt = np.r_[C["target"][i], 0.0]
        plans = {a: np.concatenate([tgt[None], cand[i][a][0], np.repeat(cand[i][a][0][-1:], 7, 0)]).astype(np.float32) for a in names}
        seed = 20261042 + i; jobs = []
        for a in names:
            base = cbff.horizon(plans[a], cbff.CFG["N_LOOK"])
            for s in SCALES:
                for j in range(n_fut):
                    p, r = mon.errors.sample(seed, 0, 0, j)
                    act = base.copy(); nz = np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
                    act[:cbff.CFG["N_LOOK"], :3] -= nz
                    jobs.append(((a, s, j), snap, act, (p * [1, 1, 0] * s, r * [0, 0, 1] * s)))
        ends = {}
        for key, fa, fb in pool.imap(_job, jobs, chunksize=4):
            ends[key] = (fa, fb)
        for a in names:
            m8 = torch.as_tensor(~cb.hand_mask(np.vstack([site[None], plans[a][1:16, :3]])), device="cuda")
            row = {"i": i, "arm": a, "risk": cand[i][a][1]}
            for rep in ("pca8", "full"):
                split = []
                for s in SCALES:
                    fr = np.stack([x for j in range(n_fut) for x in ends[(a, s, j)]])         # (2n, H, W, 3): t18, t38 interleaved
                    if rep == "pca8":
                        z = torch.as_tensor(L(fr), device="cuda").float().reshape(n_fut, 2, -1)
                    else:
                        z = dino_tokens(L.encoder, fr, "cuda").float().reshape(n_fut, 2, -1)
                    ok18, _, _ = cf.split_test_gpu(z[None, :, 0]); ok38, _, _ = cf.split_test_gpu(z[None, :, 1])
                    split.append(bool(ok18[0] & ok38[0]))
                    if s == 1.0:
                        if rep == "pca8":
                            diff = (z[:, 1].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                        else:
                            diff = (z[:, 1].reshape(n_fut, 28, 28, 384) - t0.reshape(28, 28, 384)[None]).pow(2).mean(-1)
                        o = float(((diff * m8).sum((1, 2)) / m8.sum().clamp_min(1)).max())
                hit = [k for k, x in enumerate(split) if x]
                row[f"h_{rep}"] = float(np.log2(SCALES[hit[0]])) if hit else float(np.log2(3.0)); row[f"o_{rep}"] = o
            with open(f, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print("decision", i, len(names), "chunks", flush=True)
    pool.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
