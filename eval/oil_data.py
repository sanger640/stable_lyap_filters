"""Optimiser-in-the-loop world-model data, round 1 (2026-10-16).

Why: every optimiser variant passes its own model check yet lands at 19-24% real risk offline (selection 13.8%): it
moves the chunk to where the world model was never trained (only the policy's plans and small random nudges, cf_v1)
and exploits its errors there. Standard fix in model-based control: show the model the optimiser's own proposals.
Labels are NOT used: the data is image latents under actions, exactly like closedloop_v1 / cf_v1.

propose: at N_STATES training states of cf_v1 (paired_record_dev2x episodes 0-299; the offline benchmark uses
         episodes 2000-2149, untouched), run the task-agnostic gradient optimiser (cbf_opt grad, b50 settings, the
         current world model CBF_WM) from the policy's plan + 7 smooth nudges; keep its chosen plan + the 2 other
         optimised starts with the lowest predicted disturbance -> OUT/proposals.json, OUT/selection.json.
simulate: CF_OUT=OUT CF_PLANS=OUT/proposals.json python3 eval/cf_data.py  (policy plan + the 3 proposals, 4 paired
         perturbation draws each, H15, same file format as cf_v1).

usage: CBF_WM=... OPT_ALPHA=0.5 OPT_STARTS=8 OPT_STEPS=10 OPT_COST=timed OPT_SIG0=0.987410 python3 eval/oil_data.py propose
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / os.environ.get("OIL_OUT", "results/program_b/oil_v1")
N_STATES, SEED = int(os.environ.get("OIL_N", 300)), 20261800


def propose():
    import torch
    import cbf_filter as cbff
    import cbf_opt as co
    import cem_filter as cf
    import cf_data as cd
    import closedloop_data as cl
    import inspector_data as insp
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "results/program_b/cf_v1/selection.json").read_text())       # near-topple states first
    keys = [k for k, v in sel.items() if v == "train"][:N_STATES]
    pf = OUT / "proposals.json"
    prop = json.loads(pf.read_text()) if pf.exists() else {}
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    t0, cache = time.time(), {}
    for n, k in enumerate(keys):
        if k in prop:
            continue
        e, c = map(int, k.split(":"))
        if e not in cache:
            cache.clear(); cache[e] = dict(np.load(cl.RUN / f"ep{e:03d}.npz"))
        S = cache[e]; i = int(np.flatnonzero(S["chunk_index"] == c)[0])
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0].astype(np.float32), device="cuda")
        site = sim.data.site_xpos[sim.site].copy()
        plan = np.asarray(S["plan16"][i], np.float32)
        rng = np.random.default_rng([SEED, e, c])
        starts = np.stack([plan] + [cd.nudge(plan, rng) for _ in range(7)])
        best, info = co.optimise(mon, cbf, sim, z0, site, starts, 1.0, SEED + n, fork_check=False, return_all=True)
        rest = [j for j in np.argsort(info["ob_all"]) if not np.allclose(info["all"][j], best)][:2]
        prop[k] = [best.tolist()] + [info["all"][j].tolist() for j in rest]
        if n % 20 == 0:
            pf.write_text(json.dumps(prop))
            print(f"propose {n + 1}/{len(keys)} {time.time() - t0:.0f}s change {info['change_mm']} mm sigma {info['sigma']}", flush=True)
    pf.write_text(json.dumps(prop))
    (OUT / "selection.json").write_text(json.dumps({k: "train" for k in keys}) + "\n")
    print("proposals", len(prop), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"propose": propose}[sys.argv[1]]()
