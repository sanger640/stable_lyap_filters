"""Ensemble member k of the deterministic world model (step 2 'fix recall'): fine-tuned from the closed-loop model
(closedloop_v1/model/model_US.pt) on closed-loop states + paired counterfactual states (cf_v1, and cf_v2 from fresh
cluster episodes 6000-6599 when present), with seed k (different batch order and initial input noise).
Disagreement between members = epistemic uncertainty -> trust region for the optimiser and a recall check.

usage: ENS_SEED=k python3 eval/ens_train.py   (env ENS_STEPS 3000, ENS_LR_SCALE 0.3, ENS_OUT results/program_b/ens_v1)
"""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CL = ROOT / "results/program_b/closedloop_v1"
OUT = ROOT / os.environ.get("ENS_OUT", "results/program_b/ens_v1")


def main():
    import fast_train as ft
    import phase1d_models as p1d
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    k = int(os.environ.get("ENS_SEED", os.environ.get("TASK", "0/1").split("/")[0]))
    cf_dirs = [d for d in (ROOT / "results/program_b/cf_v1/train", ROOT / "results/program_b/cf_v2/train") if d.exists()]
    cf_states = p1d.load(cf_dirs, proprio=False)
    cl_states = p1d.load([CL / "train_selected"], proprio=False)
    states = cf_states + cl_states
    n_cf = sum(len(s["z"]) for s in cf_states); n_all = sum(len(s["z"]) for s in states)
    pools = (np.arange(n_cf), np.arange(n_cf, n_all))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"member {k}: {len(cf_states)} cf states ({[d.parent.name for d in cf_dirs]}), {len(cl_states)} closed-loop", flush=True)
    ft.train(f"ENS{k}", OUT, states, sampler, seed=k, steps=int(os.environ.get("ENS_STEPS", 3000)),
             init=CL / "model/model_US.pt", lr=ft.LR * float(os.environ.get("ENS_LR_SCALE", 0.3)),
             compile_net=os.environ.get("ENS_COMPILE", "0") == "1")      # the container has no C compiler for torch.compile
    print(f"member {k} done", flush=True)


if __name__ == "__main__":
    main()
