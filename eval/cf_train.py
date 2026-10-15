"""Fine-tune the closed-loop world model (closedloop_v1/model/model_US.pt) on counterfactual data (cf_v1/train: the
policy plan + nudged plans from the same starts) mixed 50/50 with the closed-loop states it was trained on.

usage: python3 eval/cf_train.py   (env CF_STEPS 3000, CF_LR_SCALE 0.3)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CL = ROOT / "results/program_b/closedloop_v1"
CF = ROOT / "results/program_b/cf_v1"
OUT = CF / "model"


def main():
    import closedloop_train as ct
    import fast_train as ft
    import phase1d_models as p1d
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"           # G.pt as before
    ct._patch()                                                  # builds closedloop_v1/train_selected symlinks
    OUT.mkdir(parents=True, exist_ok=True)
    cl_states = p1d.load([CL / "train_selected"], proprio=False)
    cf_states = p1d.load([CF / "train"], proprio=False)
    states = cf_states + cl_states
    n_cf = sum(len(s["z"]) for s in cf_states); n_all = sum(len(s["z"]) for s in states)
    pools = (np.arange(n_cf), np.arange(n_cf, n_all))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    print("states", len(cf_states), "cf +", len(cl_states), "closed-loop; futures", n_cf, "+", n_all - n_cf, flush=True)
    steps = int(os.environ.get("CF_STEPS", 3000))
    ft.train("CF", OUT, states, sampler, steps=steps, init=CL / "model/model_US.pt",
             lr=ft.LR * float(os.environ.get("CF_LR_SCALE", 0.3)))
    print("CF done", flush=True)


if __name__ == "__main__":
    main()
