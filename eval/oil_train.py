"""Fine-tune model_CF on optimiser-in-the-loop data (oil_v1/train: the policy plan + the optimiser's proposals from the
same starts), mixed 1/3 each with the counterfactual nudges (cf_v1/train) and the closed-loop states, so the model
learns the optimiser's region without forgetting the policy's. No labels: latent dynamics only.

usage: python3 eval/oil_train.py   (env OIL_STEPS 3000, OIL_LR_SCALE 0.3, OIL_OUT results/program_b/oil_v1)
"""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CL = ROOT / "results/program_b/closedloop_v1"
CF = ROOT / "results/program_b/cf_v1"
OIL = ROOT / os.environ.get("OIL_OUT", "results/program_b/oil_v1")


def main():
    import closedloop_train as ct
    import fast_train as ft
    import phase1d_models as p1d
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    ct._patch()
    groups = [p1d.load([OIL / "train"], proprio=False), p1d.load([CF / "train"], proprio=False),
              p1d.load([CL / "train_selected"], proprio=False)]
    states = [s for g in groups for s in g]
    sizes = [sum(len(s["z"]) for s in g) for g in groups]; edges = np.cumsum([0] + sizes)
    pools = [np.arange(edges[k], edges[k + 1]) for k in range(3)]
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[k], b // 3 + (k < b % 3)) for k in range(3)])
    print("futures oil / cf / closed-loop", sizes, flush=True); (OIL / "model").mkdir(parents=True, exist_ok=True)
    ft.train("OIL", OIL / "model", states, sampler, steps=int(os.environ.get("OIL_STEPS", 3000)),
             init=CF / "model/model_CF.pt", lr=ft.LR * float(os.environ.get("OIL_LR_SCALE", 0.3)))
    print("OIL done", flush=True)


if __name__ == "__main__":
    main()
