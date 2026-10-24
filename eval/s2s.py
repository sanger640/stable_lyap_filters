"""Sim-to-sim adaptation test (results/jenga/s2s_protocol.json; the SimDist recipe, arXiv 2603.15759).
train: model_S2S = model_OIL with ONLY the latent dynamics fine-tuned (DINOv2+PCA encoder and start operator G frozen),
       3000 steps, lr x0.3, batches 1/2 shifted-physics futures (s2s_v1/train) + 1/6 each oil_v1 / cf_v1 / closed-loop.
error: latent prediction error (mean squared, per latent element) of model_OIL and model_S2S on the shifted-physics val
       futures (s2s_v1/val, evaluation episodes) and on normal-physics val futures (closedloop_v1/val): all steps, HOLD10, HOLD30.
usage: python3 eval/s2s.py train | error
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
S2S = ROOT / "results/program_b/s2s_v1"
OIL_PT = ROOT / "results/program_b/oil_v1/model/model_OIL.pt"


def train():
    import closedloop_train as ct
    import fast_train as ft
    import phase1d_models as p1d
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    ct._patch()
    P = ROOT / "results/program_b"
    groups = [p1d.load([S2S / "train"], proprio=False), p1d.load([P / "oil_v1/train"], proprio=False),
              p1d.load([P / "cf_v1/train"], proprio=False), p1d.load([P / "closedloop_v1/train_selected"], proprio=False)]
    states = [s for g in groups for s in g]
    sizes = [sum(len(s["z"]) for s in g) for g in groups]; edges = np.cumsum([0] + sizes)
    pools = [np.arange(edges[k], edges[k + 1]) for k in range(4)]
    share = np.array([3, 1, 1, 1])                                          # 1/2 shifted, 1/6 each of the original mix

    def sampler(rng, b):
        n = np.floor(b * share / share.sum()).astype(int); n[0] += b - n.sum()
        return np.concatenate([rng.choice(pools[k], n[k]) for k in range(4)])
    print("futures s2s / oil / cf / closed-loop", sizes, flush=True); (S2S / "model").mkdir(parents=True, exist_ok=True)
    ft.train("S2S", S2S / "model", states, sampler, steps=3000, init=OIL_PT, lr=ft.LR * 0.3)
    print("S2S done", flush=True)


def error():
    import torch
    import phase1d_models as p1d
    import step3_world_model as wm
    models = {"OIL": wm.WorldModel.load(OIL_PT), "S2S": wm.WorldModel.load(S2S / "model/model_S2S.pt")}
    sets = {"shifted_val": S2S / "val", "normal_val": ROOT / "results/program_b/closedloop_v1/val"}
    out = {}
    for sname, folder in sets.items():
        states = p1d.load([folder], proprio=False)
        z0k = torch.cat([s["z0k"] for s in states]).float().cuda(); tgt = torch.cat([s["z"] for s in states]).float().cuda()
        act = torch.cat([s["actions"] for s in states]).cuda()
        out[sname] = {"futures": len(z0k)}
        for mname, m in models.items():
            err = []
            for i in range(0, len(z0k), 256):
                cur = z0k[i:i + 256]; e = []
                with torch.no_grad():
                    for t in range(wm.STEPS):
                        cur = m.step(cur, act[i:i + 256, t]).float()
                        e.append(((cur - tgt[i:i + 256, t]) ** 2).mean((1, 2, 3)))
                err.append(torch.stack(e, 1).cpu())
            E = torch.cat(err).numpy()
            out[sname][mname] = {"all_steps": round(float(E.mean()), 5), "hold10": round(float(E[:, wm.HOLD10].mean()), 5),
                                 "hold30": round(float(E[:, wm.HOLD30].mean()), 5)}
    (ROOT / "results/jenga/deep_dive/s2s_error.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"train": train, "error": error}[sys.argv[1]]()
