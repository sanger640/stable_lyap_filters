"""Speed (exploratory): the filter spends ~93% of its time in world-model steps (eval/opt3 profile: 35 s per decision
locally), each a 16-step Euler integration of the flow sampler. The sampler's starting noise is drawn once per model step,
so with the same torch seed, fewer Euler steps change only the integration accuracy. Same decisions, same futures:
agreement of h (fork scale), of the 3x trigger (h_plan < min(h_hold, 2)) and of o, at EULER 16 (reference) / 8 / 4 / 2.
Benchmark v2 dev, plans: policy, hold (gripper held), G-CEM's chunk. model_OIL, 64 futures, SCALES 0.5-4.
usage: python3 eval/euler_check.py -> results/jenga/deep_dive/euler_check.json
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import bench2 as b2  # noqa: E402
from red_readout import p16_of  # noqa: E402

EUL = (16, 8, 4, 2)


def main():
    import torch
    import cbf_barrier
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    C, R = b2.load("dev"); G = json.loads((b2.OUT / "opt3_dev_g_cem.json").read_text())["rows"]
    n = int(os.environ.get("EC_N", len(G))); rows = []; secs = {e: 0.0 for e in EUL}
    for i in range(n):
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        plans = np.stack([C["plan16"][i], np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32),
                          p16_of(C, i, np.asarray(G[i]["chunk"], np.float32))])
        r = {"i": i}
        for e in EUL:
            wm.EULER = e; mon.rng = np.random.default_rng([20261129, i]); torch.manual_seed(20261129 + i)
            torch.cuda.synchronize(); t = time.time()
            h, o = cbf.score(sim, z0, plans, 64)
            torch.cuda.synchronize(); secs[e] += time.time() - t
            r[str(e)] = {"h": [float(x) for x in h], "o": [float(x) for x in o]}
        rows.append(r); print("euler", i, [r[str(e)]["h"] for e in EUL], flush=True)
    wm.EULER = 16
    out = {"decisions": len(rows), "seconds_per_score": {e: round(secs[e] / len(rows), 3) for e in EUL}, "exploratory": True}
    ref = rows
    trig = lambda h: h[0] < min(h[1], 2.0) - 1e-9
    for e in EUL[1:]:
        hs = np.array([[r["16"]["h"], r[str(e)]["h"]] for r in ref])            # (n, 2, 3)
        os_ = np.array([[r["16"]["o"], r[str(e)]["o"]] for r in ref])
        tr = np.array([[trig(r["16"]["h"]), trig(r[str(e)]["h"])] for r in ref])
        cert = np.array([[r["16"]["h"][2] >= min(r["16"]["h"][1], 2.0), r[str(e)]["h"][2] >= min(r[str(e)]["h"][1], 2.0)] for r in ref])
        out[f"euler_{e}"] = {"h_exact_agree": round(float((hs[:, 0] == hs[:, 1]).mean()), 3),
                             "h_mean_ref_vs": [round(float(hs[:, 0].mean()), 3), round(float(hs[:, 1].mean()), 3)],
                             "trigger_agree": round(float((tr[:, 0] == tr[:, 1]).mean()), 3),
                             "trigger_rate_ref_vs": [round(float(tr[:, 0].mean()), 3), round(float(tr[:, 1].mean()), 3)],
                             "gcem_cert_agree": round(float((cert[:, 0] == cert[:, 1]).mean()), 3),
                             "o_spearman": round(float(np.corrcoef(np.argsort(np.argsort(os_[:, 0].ravel())), np.argsort(np.argsort(os_[:, 1].ravel())))[0, 1]), 3),
                             "o_ratio_median": round(float(np.median(os_[:, 1].ravel() / np.maximum(os_[:, 0].ravel(), 1e-9))), 3)}
    (ROOT / "results/jenga/deep_dive/euler_check.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
