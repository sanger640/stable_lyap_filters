"""ManiSkill3 quick fork check (docs/BENCHMARK_DD.md, spec fixed before running): do outcomes SPLIT under small execution
noise along the shipped expert demos of StackPyramid-v1 / LiftPegUpright-v1?

GPU simulator, 33 parallel envs: at every 8th demo step all envs are set to the demo's saved state; env 0 replays the
next 8 demo actions exactly, envs 1-32 with 3-tap smoothed Gaussian noise; then 20 hold steps. Bad replay: an object's
final pose differs from env 0's by > 2 cm or > 20 deg. Fork: 2..30 of 32 bad.
Runs in ~/wksp/maniskill_bench/.venv.
usage: python eval/maniskill_fork_check.py StackPyramid-v1 --demos 16 --out DIR
"""
import argparse
import json
import pathlib
import time

import numpy as np

CFG = {"StackPyramid-v1": dict(h5="demos/StackPyramid-v1/motionplanning/trajectory.h5", mode="pd_joint_pos",
                               objs=("cubeA", "cubeB", "cubeC"), sig=[0.004] * 7),
       "LiftPegUpright-v1": dict(h5="demos/LiftPegUpright-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5",
                                 mode="pd_ee_delta_pose", objs=("peg",), sig=[0.025] * 3 + [0.025 / 3] * 3)}
H, HOLD, N = 8, 20, 33


def to_torch(d, n, dev):
    import torch
    if isinstance(d, dict) or hasattr(d, "keys"):
        return {k: to_torch(d[k], n, dev) for k in d.keys()}
    return torch.as_tensor(np.asarray(d), device=dev).float()[None].expand(n, -1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env"); ap.add_argument("--demos", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(pathlib.Path.home() / "wksp/maniskill_bench"))
    a = ap.parse_args()
    import gymnasium as gym
    import h5py
    import torch
    import mani_skill.envs  # noqa: F401
    c = CFG[a.env]
    env = gym.make(a.env, num_envs=N, sim_backend="physx_cuda", obs_mode="none", control_mode=c["mode"])
    env.reset(seed=0); u = env.unwrapped; dev = u.device
    f = h5py.File(pathlib.Path(a.root) / c["h5"], "r")
    rng = np.random.default_rng(20261021)
    sig = np.asarray(c["sig"])
    rows, t0 = [], time.time()

    def poses():
        return {o: (getattr(u, o).pose.p.clone(), getattr(u, o).pose.q.clone()) for o in c["objs"]}

    for di in range(a.demos):
        tr = f[f"traj_{di}"]; acts = np.asarray(tr["actions"]); T = len(acts)
        es = tr["env_states"]
        for t in range(0, T - H, H):
            st = {g: {k: to_torch(es[g][k][t], N, dev) for k in es[g].keys()} for g in es.keys()}
            u.set_state_dict(st)
            base = np.repeat(acts[t:t + H][None], N, 0).astype(np.float64)            # (N, H, d)
            nz = rng.normal(size=(N, H + 2, len(sig))) * sig
            nz = (nz[:, :-2] + nz[:, 1:-1] + nz[:, 2:]) / 3; nz[0] = 0
            base[:, :, :len(sig)] += nz
            if c["mode"] == "pd_ee_delta_pose":
                base[:, :, :6] = np.clip(base[:, :, :6], -1, 1)
                hold = base[:, -1].copy(); hold[:, :6] = 0
            else:
                hold = base[:, -1].copy()
            for k in range(H + HOLD):
                act = base[:, k] if k < H else hold
                env.step(torch.as_tensor(act, device=dev).float())
            ps = poses()
            bad = torch.zeros(N, dtype=torch.bool, device=dev); dmax = torch.zeros(N, device=dev)
            for o, (p, q) in ps.items():
                dp = (p - p[0:1]).norm(dim=1)
                dq = 2 * torch.arccos(torch.clamp((q * q[0:1]).sum(1).abs(), max=1.0)) * 180 / np.pi
                bad |= (dp > 0.02) | (dq > 20); dmax = torch.maximum(dmax, dp)
            succ = u.evaluate()["success"].cpu().numpy()
            fin = {o: torch.cat([pq[0], pq[1]], 1).cpu().numpy().round(5).tolist() for o, pq in ps.items()}   # per-replay final pose (p, q)
            nb = int(bad[1:].sum())
            rows.append({"demo": di, "t": t, "T": T, "n_bad": nb, "succ_exact": bool(succ[0]), "n_succ_noisy": int(succ[1:].sum()),
                         "dpos_max_median_mm": float(dmax[1:].median() * 1000), "final_pose": fin})
        print(di, T, [r["n_bad"] for r in rows if r["demo"] == di], round(time.time() - t0, 1), flush=True)
    nb = [r["n_bad"] for r in rows]
    fork, anyb = sum(2 <= x <= 30 for x in nb), sum(x > 0 for x in nb)
    res = {"env": a.env, "points": len(rows), "forks": fork, "points_any_bad": anyb,
           "n_bad_hist": {str(k): nb.count(k) for k in sorted(set(nb))},
           "pass": bool(fork >= 15 and fork >= 0.25 * max(anyb, 1) and fork >= 0.05 * len(rows)), "seconds": round(time.time() - t0, 1)}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.env}_rows.json").write_text(json.dumps(rows)); (out / f"{a.env}_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
