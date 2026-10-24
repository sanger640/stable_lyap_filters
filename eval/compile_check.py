"""Speed-equivalence gate for WM_COMPILE (results/jenga/compile_protocol.json). Benchmark v2 dev, plans policy / hold /
G-CEM chunk; variants ref (eager bf16), fp32 (eager, autocast off: noise floor), compiled (torch.compile bf16); same seeds.
usage: python3 eval/compile_check.py -> results/jenga/compile_result.json
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


def main():
    import torch
    import cbf_barrier
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    eager = mon.model.net; compiled = torch.compile(eager); ac = torch.autocast
    OMAX = cbff.CFG["O_MAX"]
    C, R = b2.load("dev"); G = json.loads((b2.OUT / "opt3_dev_g_cem.json").read_text())["rows"]
    V = ("ref", "fp32", "compiled"); rows = []; secs = {v: [] for v in V}
    for i in range(len(G)):
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        plans = np.stack([C["plan16"][i], np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32),
                          p16_of(C, i, np.asarray(G[i]["chunk"], np.float32))])
        r = {"i": i}
        for v in V:
            mon.model.net = compiled if v == "compiled" else eager
            torch.autocast = (lambda *a, **k: ac(*a, **{**k, "enabled": False})) if v == "fp32" else ac
            mon.rng = np.random.default_rng([20261130, i]); torch.manual_seed(20261130 + i)
            torch.cuda.synchronize(); t = time.time()
            h, o = cbf.score(sim, z0, plans, 64)
            torch.cuda.synchronize(); secs[v].append(time.time() - t)
            r[v] = {"h": [float(x) for x in h], "o": [float(x) for x in o]}
        torch.autocast = ac; mon.model.net = eager
        rows.append(r); print("cc", i, [r[v]["h"] for v in V], flush=True)
    trig = lambda x: x["h"][0] < min(x["h"][1], 2.0) - 1e-9
    cert = lambda x: x["h"][2] >= min(x["h"][1], 2.0) - 1e-9 and x["o"][2] <= OMAX
    rk = lambda a: np.argsort(np.argsort(a))
    out = {"decisions": len(rows), "seconds_per_score_median": {v: round(float(np.median(secs[v][1:])), 3) for v in V},
           "compile_first_call_s": round(secs["compiled"][0], 1)}
    for v in ("fp32", "compiled"):
        out[v] = {"trigger_agree": round(float(np.mean([trig(r[v]) == trig(r["ref"]) for r in rows])), 4),
                  "cert_agree": round(float(np.mean([cert(r[v]) == cert(r["ref"]) for r in rows])), 4),
                  "h_exact_agree": round(float(np.mean([a == b for r in rows for a, b in zip(r[v]["h"], r["ref"]["h"])])), 4),
                  "o_spearman": round(float(np.corrcoef(rk([x for r in rows for x in r[v]["o"]]), rk([x for r in rows for x in r["ref"]["o"]]))[0, 1]), 4),
                  "trigger_rate": round(float(np.mean([trig(r[v]) for r in rows])), 4)}
    out["ref_trigger_rate"] = round(float(np.mean([trig(r["ref"]) for r in rows])), 4)
    thr = {k: min(0.95, out["fp32"][k]) for k in ("trigger_agree", "cert_agree")}
    out["thresholds"] = thr
    out["PRIMARY_pass"] = bool(all(out["compiled"][k] >= thr[k] for k in thr))
    (ROOT / "results/jenga/compile_result.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results/jenga/deep_dive/compile_rows.json").write_text(json.dumps(rows))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
