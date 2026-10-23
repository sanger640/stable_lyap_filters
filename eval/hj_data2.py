"""HJ v2 stage B data (docs/HJ_V2.md): for every saved decision of logged closed-loop episodes -> the latent z (float16),
proprioception (gripper site xyz, gripper closed) and the hold-plan fork statistics of eval/hj_margin.hold_stats at
HJ_SCALES (raw, so any stage-A margin can be computed later). Label-free (outcome flags are copied only for evaluation).
env: HJ_RUNS = space-separated run dirs under results/program_b/policy_rollouts (e.g. "paired_record_s4 paired_cbfg_s4"),
     HJ_SCALES = e.g. "2,4" (default all of hj_margin.SC), TASK = i/n (cluster array shard).
usage: python3 eval/hj_data2.py   -> results/jenga/hj2/data/<run>/ep<e>.npz (resumable)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
PR = ROOT / "results/program_b/policy_rollouts"
OUT = ROOT / "results/jenga/hj2/data"


def main():
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import hj_margin as hm
    import inspector_data as insp
    import step3_world_model as wm
    if os.environ.get("HJ_SCALES"):
        hm.SC = tuple(float(x) for x in os.environ["HJ_SCALES"].split(","))
    i, n = (int(x) for x in os.environ.get("TASK", "0/1").split("/"))
    files = sorted(f for run in os.environ["HJ_RUNS"].split() for f in (PR / run).glob("ep*.npz"))[i::n]
    mon = cf.Monitor(); sim = insp._sim()
    for f in files:
        out = OUT / f.parent.name / f.name
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with np.load(f) as z:
            S = {k: z[k] for k in z.files}
        L = json.loads(f.with_suffix(".json").read_text())["log"]
        kt = next((c["chunk"] for c in L if c.get("topple_started")), -1)
        ep = int(f.stem[2:]); Z, P, ST = [], [], []
        for d in range(len(S["chunk_index"])):
            k = int(S["chunk_index"][d])
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
            mon.rng = np.random.default_rng([20261121, ep, k])
            z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
            st = hm.hold_stats(mon, cbff, wm, z0, S["target"][d])
            Z.append(z0.cpu().numpy().astype(np.float16))
            P.append(np.r_[sim.data.site_xpos[sim.site], 1.0 if float(S["gripper"][d]) < 50 else -1.0].astype(np.float32))
            ST.append([[st[s][e][q] for e in ("10", "30") for q in ("ok", "D", "gain", "spread")] for s in hm.SC])
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, z=np.stack(Z), proprio=np.stack(P), stats=np.asarray(ST, np.float32),
                            scales=np.asarray(hm.SC, np.float32), chunk_index=S["chunk_index"], topple_chunk=kt,
                            stat_names=np.array(["ok10", "D10", "gain10", "spread10", "ok30", "D30", "gain30", "spread30"]))
        tmp.rename(out)
        print("hjdata", f.parent.name, ep, len(Z), flush=True)


if __name__ == "__main__":
    main()
