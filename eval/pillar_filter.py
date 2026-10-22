"""Pillar task closed-loop safety filter: the universality test pre-registered in docs/OPT_FILTER_REDESIGN.md
("the frozen family and ranges, with no retuning, on a second task"); spec in docs/PILLAR_TASK.md "Closed-loop filter".

Per decision: encode the side camera (DINOv2 -> PCA 8, robot patches zeroed, eval/pillar_wm.py), imagine the policy's
chunk + 22 hold under 32 draws of the real execution noise, score d = largest imagined end-of-hold latent distance from
the imagined zero-noise replay (the v2 score that passed the world-model gate). Trigger: d > TAU (label-free: 90th
percentile of d over the policy's own decisions on the training seeds). Arms:
  record : the policy alone (the trigger is still computed and logged, never acted on)
  hold   : trigger -> hold still for one chunk (current target, the policy's first gripper command), then re-plan
  g      : trigger -> G-CEM over theta = (speed, pause, amplitude, back-off) with Jenga's frozen family, ranges and CEM
           settings (eval/opt3.py); certified = d <= TAU on 32 fresh draws; cheapest certified (opt2 cost), else hold if
           certified, else the lowest-d final.
usage: python3 eval/pillar_filter.py tau ; run <arm> <first> <n> [procs] ; analyse
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PILLAR_HALF_W", "0.008"); os.environ.setdefault("PILLAR_PERC", "0.002")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import pillar_wm as W  # noqa: E402

OUT = ROOT / "results/pillar/filter"
N_IMAG, TAU_Q, NOISE_SCALE = 32, 90, 2.0
# Formulation G, frozen (eval/opt3.py CFG / LO / HI / IDENTITY, opt2 ETA)
CEM = {"POP": 16, "ITERS": 3, "ELITES": 6, "FINAL": 4, "SD": np.array([0.4, 3.0, 0.4, 12.0])}
LO, HI, IDENTITY, B_MIN_MOVE_MM, ETA = np.zeros(4), np.array([1.0, 8.0, 1.0, 30.0]), np.array([1.0, 0.0, 1.0, 0.0]), 2.0, 4.0


def transform(p16, theta):
    """eval/opt3.py transform (G, gripper on the policy's clock), numpy: p16 (16, 4), theta (S, 4) -> (S, 16, 4)."""
    v, k, s, b = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4] / 1000
    t = np.arange(16, dtype=np.float64)[None]
    tau = np.clip(v * np.maximum(t - k, 0), 0, 15)
    i = np.clip(np.floor(tau), 0, 14).astype(int); w = (tau - i)[..., None]
    xyz = p16[:, :3].astype(np.float64)
    q = xyz[0] + s[..., None] * (xyz[i] * (1 - w) + xyz[i + 1] * w - xyz[0])
    net = xyz[15] - xyz[0]; nrm = np.linalg.norm(net)
    if nrm * 1000 >= B_MIN_MOVE_MM:
        q = q - b[..., None] * (np.clip(t, None, 8) / 8)[..., None] * (net / nrm)
    return np.concatenate([q, np.broadcast_to(p16[None, :, 3:], (len(theta), 16, 1))], -1).astype(np.float32)


def cost(q16, p16):
    """opt2 Problem.cost on the xyz change of steps 1-8 (mm^2 + ETA * second differences^2) / 100."""
    mm = (q16[1:9, :3] - p16[1:9, :3]) * 1000
    sd = np.diff(np.vstack([np.zeros((1, 3)), mm]), 2, axis=0)
    return float((mm ** 2).sum() + ETA * (sd ** 2).sum()) / 100


class Scorer:
    def __init__(self):
        import step3_world_model as wm
        import uncertainty_monitor_dev as um
        W._patch_wm()
        self.model = wm.WorldModel.load(W.OUT / "model_PW.pt")
        self.snippets = np.load(um.BENCH)["snippets"]
        self.lat = None

    def latent(self, sim, cam):
        self.lat = self.lat or W.Latent()
        f, m = cam()
        return self.lat(f[None], W.patch_keep(m[None]))[0]

    def d(self, z0, p0, chunks, nz):
        """chunks (C, 8, 4) absolute targets, nz (N, 8, 3) noise -> d (C,) with the zero-noise replay as reference."""
        import torch
        C, N = len(chunks), len(nz)
        nzz = np.concatenate([np.zeros((1, 8, 3)), nz])                                     # (N + 1, 8, 3)
        acts = np.concatenate([chunks, np.repeat(chunks[:, -1:], W.HOLD, 1)], 1)            # (C, 30, 4)
        acts = np.repeat(acts[:, None], N + 1, 1).copy()
        acts[:, :, :8, :3] -= nzz[None]; acts[:, :, 8:, :3] -= nzz[None, :, -1:]
        acts = acts.reshape(C * (N + 1), 30, 4)
        inp = torch.as_tensor(W.model_inputs(acts, np.repeat(p0[None], len(acts), 0)), device="cuda")
        z = torch.as_tensor(z0, device="cuda").float()[None].expand(len(acts), -1, -1, -1).contiguous()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(W.STEPS):
                z = self.model.step(z, inp[:, t]).float()
        e = z.reshape(C, N + 1, -1)
        return (e[:, 1:] - e[:, :1]).norm(dim=2).max(1).values.cpu().numpy()

    def noise(self, rng):
        return self.snippets[rng.integers(len(self.snippets), size=N_IMAG)] * NOISE_SCALE


def stage_tau():
    """TAU = 90th percentile of d of the policy's own chunk at the training decisions (label-free)."""
    sc = Scorer(); rng = np.random.default_rng(20261024); ds = []
    for f in sorted((W.ENC / "train").glob("*.npz")):
        d = np.load(f); src = np.load(W.DATA / "train" / f.name.split("_", 1)[0] / f.name.split("_", 1)[1])
        ds.append(float(sc.d(d["z"][0, 0].astype(np.float32), src["proprio"][0], src["actions"][0][None, :8], sc.noise(rng))[0]))
    out = {"tau": float(np.percentile(ds, TAU_Q)), "quantile": TAU_Q, "decisions": len(ds),
           "d_median": float(np.median(ds)), "d_p99": float(np.percentile(ds, 99))}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tau.json").write_text(json.dumps(out, indent=1)); print(out)


def make_hook(arm, sc, cam, tau, log, seed):
    rng = np.random.default_rng([20261025, seed])

    def hook(sim, plan, k):
        t0 = time.time()
        p16 = np.vstack([np.r_[sim.target, plan[0, 3]][None], plan[:15]]).astype(np.float32)   # anchored at sim.target (A4)
        z0, p0 = sc.latent(sim, cam), W.proprio(sim)
        nz = sc.noise(rng)
        d_pol = float(sc.d(z0, p0, plan[None, :8], nz)[0])
        row = {"k": k, "d": round(d_pol, 3), "trig": bool(d_pol > tau)}
        out = plan[:8]
        hold = np.repeat(np.r_[sim.target, plan[0, 3]][None], 8, 0).astype(np.float32)
        if row["trig"] and arm == "hold":
            out, row["kind"] = hold, "hold"
        elif row["trig"] and arm == "g":
            mu, sd, pool = IDENTITY.copy(), CEM["SD"].copy(), []
            for it in range(CEM["ITERS"]):
                T = np.clip(mu + sd * rng.standard_normal((CEM["POP"], 4)), LO, HI)
                Q = transform(p16, T)
                dd = sc.d(z0, p0, Q[:, 1:9], nz)                                            # common random numbers
                key = [(0, cost(Q[c], p16)) if dd[c] <= tau else (1, float(dd[c])) for c in range(len(T))]
                order = sorted(range(len(T)), key=lambda c: key[c])
                el = T[order[:CEM["ELITES"]]]
                mu, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (HI - LO))
                pool += [(key[c], T[c]) for c in order[:CEM["FINAL"]]]
            pool.sort(key=lambda x: x[0]); finals = np.array([th for _, th in pool[:CEM["FINAL"]]])
            Q = transform(p16, finals)
            dc = sc.d(z0, p0, np.concatenate([Q[:, 1:9], hold[None]]), sc.noise(rng))       # certificate: fresh draws
            ok = dc <= tau; costs = np.array([cost(q, p16) for q in Q])
            if ok[:-1].any():
                j = int(np.flatnonzero(ok[:-1])[np.argmin(costs[ok[:-1]])]); out, row["kind"] = Q[j, 1:9], "certified"
                row["theta"] = [round(float(x), 3) for x in finals[j]]
            elif ok[-1]:
                out, row["kind"] = hold, "fallback_hold"
            else:
                j = int(np.argmin(dc[:-1])); out, row["kind"] = Q[j, 1:9], "fallback_best"
                row["theta"] = [round(float(x), 3) for x in finals[j]]
            row["d_out"] = round(float(dc[j] if row["kind"] != "fallback_hold" else dc[-1]), 3)
        row["sec"] = round(time.time() - t0, 2); log.append(row)
        return np.asarray(out, np.float32)
    return hook


_SC = None


def episode_job(args):
    global _SC
    arm, seed = args
    import pillar_task as P
    f = OUT / arm / f"s{seed}.json"
    if f.exists():
        return json.loads(f.read_text())["outcome"]
    _SC = _SC or Scorer()
    tau = json.loads((OUT / "tau.json").read_text())["tau"]
    sim = P.PillarSim(); P.reset(sim, seed); cam = W.Cam(sim); log = []
    oc, _, chunks = P.run_closed_loop(sim, seed, noise_scale=NOISE_SCALE, plan_hook=make_hook(arm, _SC, cam, tau, log, seed))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"seed": seed, "arm": arm, "outcome": oc, "chunks": len(chunks), "log": log}))
    return oc


def stage_run(arm, first, n, procs=4):
    import multiprocessing as mp
    jobs = [(arm, s) for s in range(int(first), int(first) + int(n))]
    with mp.get_context("spawn").Pool(int(procs)) as pool:
        for i, oc in enumerate(pool.imap_unordered(episode_job, jobs), 1):
            if i % 20 == 0:
                print(arm, i, flush=True)
    print("done", arm, flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "tau":
        stage_tau()
    elif st == "run":
        stage_run(*sys.argv[2:])
