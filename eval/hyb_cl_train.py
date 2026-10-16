"""Hybrid world model for the closed-loop setting: frozen deterministic stream = the counterfactual-tuned closed-loop
model (cf_v1/model/model_CF.pt) + residual flow-matching head (phase2_hybrid recipe, unchanged), trained on the
policy's closed-loop states (closedloop_v1/train_selected) and the paired counterfactual states (cf_v1/train), 50/50.
The three-block hybrid (HYB_T1) halved the plain model's false alarms on the final monitor panels (4.6% vs 11.6%).

usage: HYB_STEPS=8000 FAST_TRAIN_Z_ON_CPU=1 python3 eval/hyb_cl_train.py
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CL = ROOT / "results/program_b/closedloop_v1"
CF = ROOT / "results/program_b/cf_v1"


def main():
    import closedloop_train as ct
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import phase2_models as pm
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    ct._patch()
    folders = [CF / "train", CL / "train_selected"]
    names = [str(p) for f in folders for p in sorted(Path(f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    states = p1d.load(folders, proprio=False)
    assert len(names) == len(states)
    n_cf = len(list((CF / "train").glob("ep*.npz")))
    # pools for the hybrid sampler: "rare" = counterfactual futures, the rest = closed-loop futures (50/50 batches)
    modes = {n: (list(range(len(s["z"]))) if k < n_cf else [-1]) for k, (n, s) in enumerate(zip(names, states))}
    hy.OUT = CF / "model"
    (hy.OUT / "modes.json").write_text(json.dumps(modes))
    pm.load_train = lambda: (states, names)
    hy.STEPS = int(os.environ.get("HYB_STEPS", 8000))
    print("states", len(states), "(cf", n_cf, ") steps", hy.STEPS, flush=True)
    hy.train(name="HYB_CF", det_name="CF")


if __name__ == "__main__":
    main()
