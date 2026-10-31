"""VIS1 stage V1 (results/jenga/vis1_protocol.json), offline: does the safety value still rank danger when the filter's camera
sees a different look? CF3's 4679 fresh decisions (irrm1 rows, as eval/irrm_gate.py decisions()) are re-rendered from their
saved sim states under each look of cem_filter.VIS_SHIFT (none / light / colour / camera), encoded with the frozen DINOv2+PCA
encoder, and graded with V(z, g) of value_irr8_alea.pt exactly as gate A: AUC(-V, pre-topple 1-3 chunks vs safe), V < 0 on
safe gripper-closed / open. 'stored' = the stored latents (gate A's AUC 0.809); 'none' re-renders them (differences reported).
Also reported: the latent shift, median ||z_look - z_none|| / median ||z_none - mean z_none||.
usage: python3 eval/vis1_offline.py -> results/jenga/vis1_offline.json"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
LOOKS = ("none", "light", "colour", "camera")


def main():
    import torch
    from sklearn.metrics import roc_auc_score
    import cem_filter as cf
    import cf_conf as CC
    import inspector_data as insp
    import lsf as L
    from step3_data import Latent
    fs = sorted((ROOT / "results/jenga").glob("irrm1_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("run", "ep", "i", "kt", "k", "closed")}
    sim, lat, vf = insp._sim(), Latent(), L.ValueFn("value_irr8_alea.pt")
    N = int(os.environ.get("VIS1_LIMIT", len(R["ep"]))); R = {k: v[:N] for k, v in R.items()}; Z = {lk: np.zeros((N, 28, 28, 8), np.float16) for lk in LOOKS}; G = np.zeros((N, 4), np.float32)
    zstored = np.zeros((N, 28, 28, 8), np.float16)
    mons = {}
    for lk in LOOKS:                                   # one renderer per look (the camera look is set when it is created)
        os.environ["VIS_SHIFT"] = "" if lk == "none" else lk

        class M:
            renderer = None
        mons[lk] = M(); cf.Monitor.frame.__get__(mons[lk])(sim)
    cache, buf = {}, {lk: [] for lk in LOOKS}

    def flush():
        for lk in LOOKS:
            if buf[lk]:
                idx, frames = zip(*buf[lk]); Z[lk][list(idx)] = lat(np.stack(frames)); buf[lk] = []
    for n in range(N):
        run_ = CC.RUNS[int(R["run"][n])]; stem = f"ep{int(R['ep'][n])}"; i = int(R["i"][n])
        if (run_, stem) not in cache:
            cache.clear()
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                cache[(run_, stem)] = {k: d[k] for k in ("state", "ctrl", "target", "gripper", "plan16")}
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                cache[(run_, stem)]["z"] = d["z"]
        S = cache[(run_, stem)]
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        G[n] = np.r_[S["target"][i], S["plan16"][i][0, 3]]; zstored[n] = S["z"][i]
        for lk in LOOKS:
            os.environ["VIS_SHIFT"] = "" if lk == "none" else lk
            buf[lk].append((n, cf.Monitor.frame.__get__(mons[lk])(sim)))
        if len(buf["none"]) == 64:
            flush()
        if n % 500 == 0:
            print("vis1", n, N, flush=True)
    flush(); os.environ["VIS_SHIFT"] = ""
    kt, k, cl = R["kt"], R["k"], R["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0; ok = pre | safe
    zn = Z["none"].astype(np.float32).reshape(N, -1); spread = float(np.median(np.linalg.norm(zn - zn.mean(0), axis=1)))
    out = {"decisions": N, "value": "value_irr8_alea.pt", "looks": {}}
    for lk in LOOKS + ("stored",):
        V = np.zeros(N); Zl = zstored if lk == "stored" else Z[lk]
        with torch.no_grad():
            for b in range(0, N, 256):
                V[b:b + 256] = vf.V(torch.as_tensor(Zl[b:b + 256], device="cuda").float(),
                                    torch.as_tensor(G[b:b + 256], device="cuda")).cpu().numpy().ravel()
        zl = Zl.astype(np.float32).reshape(N, -1)
        if lk == "none":
            Vnone = V.copy()
        out["looks"][lk] = {"auc_minusV": round(float(roc_auc_score(pre[ok], -V[ok])), 4),
                            "V_below0": {"pre": round(float((V[pre] < 0).mean()), 4), "safe_closed": round(float((V[safe & cl] < 0).mean()), 4),
                                         "safe_open": round(float((V[safe & ~cl] < 0).mean()), 4)},
                            "V_median": {"pre": round(float(np.median(V[pre])), 4), "safe": round(float(np.median(V[safe])), 4)},
                            "latent_shift_rel": round(float(np.median(np.linalg.norm(zl - zn, axis=1))) / spread, 4)}
        if lk == "stored":
            out["looks"][lk]["none_vs_stored"] = {"max_abs_dz": float(np.abs(Z["none"].astype(np.float32) - zstored.astype(np.float32)).max()),
                "mean_abs_dz": float(np.abs(Z["none"].astype(np.float32) - zstored.astype(np.float32)).mean()), "max_abs_dV": float(np.abs(Vnone - V).max()),
                "median_abs_dV": float(np.median(np.abs(Vnone - V)))}
        print(lk, out["looks"][lk], flush=True)
    (ROOT / "results/jenga/vis1_offline.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
