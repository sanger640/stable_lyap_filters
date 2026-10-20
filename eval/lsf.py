"""Latent safety filter (LSF) baseline on Jenga — docs/LATENT_SAFETY_FILTER.md (protocol fixed before this code).

Stages:
  data        (CPU) labelled latents from the training splits of closedloop_v1, cf_v1, cf_v2, oil_v1: every future
              step's latent (28x28x8), proprioception (current target xyz + gripper) and a sticky failure label
              (neighbour toppled, eval/fresh_fragility_panel rule); all failures + 3x as many safe latents (seeded);
              validation splits -> a held-out set. Also reset latents for imagination (z0 of every training state).
  classifier  (GPU) l(z, g): conv trunk + MLP, tanh output, hinge loss with margin DELTA (labels: safe +, failed -)
  value       (GPU) DDPG in the world model's imagination (model_OIL): critic V/Q, actor = safety policy (bounded target
              change), discounted reachability target y = (1-g) l + g min(l, Q'(z', pi'(z'))), g = 0.9999
usage: python3 eval/lsf.py data|classifier|value
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/program_b/lsf_v1"
SOURCES = ["closedloop_v1", "cf_v1", "cf_v2", "oil_v1"]
SAFE_PER_FAIL, SEED = 3, 20261020


def labels_for(path):
    """-> latents (K, 19, 28, 28, 8) fp16, proprio (K, 19, 4), fail (K, 19) bool (sticky), z0, start target."""
    from fresh_fragility_panel import neighbour_tilts, TOPPLE_DEG
    d = np.load(path)
    tr, fi, acts = d["traces"], d["frame_index"], d["actions"]
    elig = d["start_tilt"] < TOPPLE_DEG
    K = len(tr); fail = np.zeros((K, len(fi)), bool); prop = np.zeros((K, len(fi), 4), np.float32)
    for k in range(K):
        f = np.array([np.any((neighbour_tilts(tr[k][min(i, len(tr[k]) - 1)]) >= TOPPLE_DEG) & elig) for i in fi])
        fail[k] = np.maximum.accumulate(f)
        prop[k] = acts[k][np.minimum(fi, len(acts[k]) - 1)]           # the target commanded at that control step
    return d["z"], prop, fail, d["z0"], acts[:, 0]


def stage_data():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for split in ("train", "val"):
        Z, P, Y, R0, RP = [], [], [], [], []
        for src in SOURCES:
            files = [f for f in sorted(glob.glob(str(ROOT / f"results/program_b/{src}/{split}/ep*.npz"))) if not f.endswith(".tmp.npz")]
            for f in files:
                z, prop, fail, z0, a0 = labels_for(f)
                flat_fail = fail.reshape(-1)
                idx_f = np.flatnonzero(flat_fail); idx_s = np.flatnonzero(~flat_fail)
                keep_s = rng.choice(idx_s, min(len(idx_s), SAFE_PER_FAIL * max(len(idx_f), 1)), replace=False) if len(idx_s) else idx_s
                keep = np.concatenate([idx_f, keep_s])
                Z.append(z.reshape(-1, 28, 28, 8)[keep]); P.append(prop.reshape(-1, 4)[keep]); Y.append(flat_fail[keep])
                R0.append(z0); RP.append(a0[0])
            print(split, src, len(files), "files", flush=True)
        Z, P, Y = np.concatenate(Z), np.concatenate(P), np.concatenate(Y)
        np.save(OUT / f"{split}_z.npy", Z.astype(np.float16)); np.save(OUT / f"{split}_prop.npy", P); np.save(OUT / f"{split}_fail.npy", Y)
        np.save(OUT / f"{split}_reset_z.npy", np.stack(R0).astype(np.float16)); np.save(OUT / f"{split}_reset_prop.npy", np.stack(RP))
        json.dump({"latents": int(len(Y)), "failed": int(Y.sum()), "resets": len(R0)}, open(OUT / f"{split}_summary.json", "w"))
        print(split, "latents", len(Y), "failed", int(Y.sum()), "resets", len(R0), flush=True)


if __name__ == "__main__":
    {"data": stage_data}[sys.argv[1]]()
