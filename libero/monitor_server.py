"""SafeLIBERO safety filter server (Jenga env, GPU): fork-monitor trigger + formulation G (G-CEM) or hold, the same
rules as the Jenga filter (eval/cbf_filter.py FIX=g3, eval/opt3.py, eval/opt2.py), on the LIBERO world model.

Per decision the client sends the agentview frame (JPEG q95 at 392 px, as the training data), the policy's 10-step
delta chunk, the end-effector position, the arm link positions (proprioception) and the scenario's camera key.
  trigger: act iff NOT (h(chunk) >= min(h_hold, log2 3) and o(chunk) <= O_MAX)        (64 futures, base model)
  hold:    zero arm deltas for the chunk, gripper column untouched (hold-and-replan)
  gcem:    CEM over theta = (v, k, s, b) of THIS chunk (pop 16 x 3 iterations, 6 elites, 4 finals, 32 futures, common
           random numbers), certificate = every model (base + 4 ensemble members, 64 fresh futures) passes; cheapest
           certified final, else hold if certified, else the final with the largest h (smallest o)
  cost:    executed part only (first 5 of 10 steps, as Jenga's 8 of 16): sum over steps of squared mm deviation of the
           cumulative position from the policy's + ETA x squared second differences, / 100
No object state, no labels, no task goal.

usage: python3 libero/monitor_server.py [--port 6100]
"""
import argparse
import io
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "libero"), str(ROOT / "eval"), str(ROOT / "src")]
import g_transform as G  # noqa: E402
import wm_libero as W  # noqa: E402

CFG = {"N_TRIG": 64, "N_SEARCH": 32, "N_CERT": 64, "CEM_POP": 16, "CEM_ITERS": 3, "CEM_ELITES": 6, "CEM_FINAL": 4,
       "CEM_SD": [0.4, 3.0, 0.4, 12.0], "ETA": 4.0, "EXEC": 5, "HS": W.H_CAP}


def arr(x):
    x = np.ascontiguousarray(x); return {"b": x.tobytes(), "d": str(x.dtype), "s": list(x.shape)}


def unarr(m):
    return np.frombuffer(m["b"], m["d"]).reshape(m["s"]).copy()


class Filter:
    def __init__(self):
        import torch
        import step3_world_model as wm
        W._patch_wm()
        self.torch = torch
        self.lat = W.Latent()
        self.base = wm.WorldModel.load(W.OUT / "model_L.pt")
        self.certs = [self.base] + [wm.WorldModel.load(W.OUT / f"model_ENS{k}.pt") for k in range(4)]
        self.cam = json.loads((ROOT / "results/libero/camera.json").read_text())
        self.O_MAX = json.loads((W.OUT / "calibration.json").read_text())["O_MAX"]

    def ok(self, h, o, h_hold):
        return (np.asarray(h) >= min(h_hold, CFG["HS"])) & (np.asarray(o) <= self.O_MAX)

    def cost(self, plan, p0):
        D = (np.cumsum(plan[:CFG["EXEC"], :3], 0) - np.cumsum(p0[:CFG["EXEC"], :3], 0)) * G.MM_PER_UNIT
        sd = np.diff(np.vstack([np.zeros((1, 3)), D]), 2, axis=0)
        return float((D ** 2).sum() + CFG["ETA"] * (sd ** 2).sum()) / 100

    def decide(self, req):
        t0 = time.time()
        from PIL import Image
        frame = np.asarray(Image.open(io.BytesIO(req["jpg"])).convert("RGB"))[None]
        z0 = self.torch.as_tensor(self.lat(frame)[0], device="cuda").float()
        chunk, eef, links = unarr(req["chunk"]).astype(np.float64), unarr(req["eef"]), unarr(req["links"])
        cam, seed = self.cam[req["scenario"]], int(req["seed"])
        hold = np.zeros_like(chunk); hold[:, 6] = chunk[0, 6]
        sc = lambda m, plans, n, s: W.score_plans(m, z0, plans, eef, links, cam, n, s)
        h, o = sc(self.base, [chunk, hold], CFG["N_TRIG"], seed)
        info = {"h_nom": float(h[0]), "o_nom": float(o[0]), "h_hold": float(h[1]), "o_hold": float(o[1])}
        trig = not bool(self.ok(h[:1], o[:1], h[1])[0])
        info["trigger"] = trig
        if not trig or req["arm"] == "observe":
            return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
        if req["arm"] == "hold":
            out = np.zeros_like(chunk); out[:, 6] = chunk[:, 6]
            return out, {**info, "intervened": True, "kind": "hold", "seconds": round(time.time() - t0, 2)}
        return self.gcem(chunk, hold, float(h[1]), sc, seed, info, t0)

    def gcem(self, p0, hold, h_hold, sc, seed, info, t0):
        rng = np.random.default_rng(seed + 1)
        mu, sd = G.IDENTITY.copy(), np.array(CFG["CEM_SD"]); pool = []
        for it in range(CFG["CEM_ITERS"]):
            T = np.clip(mu + sd * rng.standard_normal((CFG["CEM_POP"], 4)), G.LO, G.HI)
            plans = [G.transform(p0, th) for th in T]
            h, o = sc(self.base, plans, CFG["N_SEARCH"], seed + 2)                 # common random numbers across iterations
            ok = self.ok(h, o, h_hold); cost = np.array([self.cost(p, p0) for p in plans])
            key = [(0 if ok[c] else 1, cost[c] if ok[c] else -h[c], 0 if ok[c] else o[c]) for c in range(len(T))]
            order = sorted(range(len(T)), key=lambda c: key[c])
            el = T[order[:CFG["CEM_ELITES"]]]
            mu, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (G.HI - G.LO))
            pool += [(key[c], T[c]) for c in order[:CFG["CEM_FINAL"]]]
        pool.sort(key=lambda x: x[0])
        finals = [th for _, th in pool[:CFG["CEM_FINAL"]]]
        plans = [G.transform(p0, th) for th in finals] + [hold]
        ok = None
        for k, m in enumerate(self.certs):                                          # fresh draws; every model must pass
            h, o = sc(m, plans, CFG["N_CERT"], seed + 3)
            okk = self.ok(h, o, h_hold); ok = okk if ok is None else ok & okk
            if k == 0:
                H, O = h, o
        costs = np.array([self.cost(p, p0) for p in plans[:-1]]); okc = ok[:-1]
        if okc.any():
            j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); kind = "certified"
        elif ok[-1]:
            j, kind = -1, "fallback_hold"
        else:
            j = int(np.lexsort((O[:-1], -H[:-1]))[0]); kind = "fallback_best"
        out = plans[j]
        if kind == "fallback_hold":
            out = np.zeros_like(p0); out[:, 6] = p0[:, 6]                           # hold keeps the policy's gripper clock
        theta = None if j < 0 else [round(float(x), 3) for x in finals[j]]
        a = np.linalg.norm(np.cumsum(p0[:CFG["EXEC"], :3], 0)[-1]); b = np.linalg.norm(np.cumsum(out[:CFG["EXEC"], :3], 0)[-1])
        return out, {**info, "intervened": True, "kind": kind, "theta": theta, "n_certified": int(okc.sum()),
                     "search_certified": int(sum(k[0] == 0 for k, _ in pool)), "progress": round(float(b / a), 3) if a > 1e-6 else 1.0,
                     "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6100); a = ap.parse_args()
    import threading
    f = Filter(); lock = threading.Lock()

    def serve(conn):
        try:
            while True:
                req = conn.recv()
                with lock:                                   # one decision on the GPU at a time
                    out, info = f.decide(req)
                conn.send({"chunk": arr(np.asarray(out, np.float64)), "info": info})
        except (EOFError, ConnectionResetError, OSError):
            conn.close()
    with Listener(("localhost", a.port), authkey=b"libero-filter") as lis:
        print("monitor server ready on", a.port, "O_MAX", f.O_MAX, flush=True)
        while True:
            threading.Thread(target=serve, args=(lis.accept(),), daemon=True).start()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
