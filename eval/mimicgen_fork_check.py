"""MimicGen StackThree_D1 quick fork check (docs/BENCHMARK_DD.md, spec fixed before running): along the shipped demos,
do outcomes SPLIT under small execution noise?

At every 8th demo step the saved simulator state is restored; 33 replays of the next 8 demo actions (exact + 32 with the
ToolHang noise: 3-tap smoothed Gaussian, sigma 0.05 xyz / 0.05/3 rot on the normalised OSC delta) + 20 hold steps.
Strict physical label per replay, relative to the exact replay's end: BAD = some cube ends > 2 cm lower or tilted
> 45 deg more (it fell / the stack collapsed); GOOD = every cube within 1 cm of the exact replay's end. Fork = >= 2 BAD
and >= 2 GOOD. Also recorded: the env's success flag at the end. Zero-noise control (--scale 0) must give no forks.
Runs in ~/wksp/mimicgen_bench/.venv.  usage: python eval/mimicgen_fork_check.py --demos 32 --procs 16 --out DIR [--scale 1]
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

ROOT = pathlib.Path.home() / "wksp/mimicgen_bench"
H5 = ROOT / "data/core/stack_three_d1.hdf5"
H, HOLD, N = 8, 20, 33
CUBES = ("cubeA", "cubeB", "cubeC")


def make_env():
    import h5py
    import mimicgen  # noqa: F401  (registers StackThree_D1)
    import robosuite
    ea = json.loads(h5py.File(H5, "r")["data"].attrs["env_args"])
    kw = dict(ea["env_kwargs"]); kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    return robosuite.make(ea["env_name"], **kw)


def cube_pose(env):
    out = []
    for c in CUBES:
        bid = env.sim.model.body_name2id(getattr(env, c).root_body)
        p = env.sim.data.body_xpos[bid].copy(); R = env.sim.data.body_xmat[bid].reshape(3, 3)
        out.append((p, float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))))  # tilt of the cube's z axis from vertical
    return out


def run_demo(args):
    di, scale = args
    import h5py
    env = make_env(); env.reset()
    d = h5py.File(H5, "r")["data"][f"demo_{di}"]
    import os
    import robosuite
    xml = d.attrs["model_file"].replace("/home/amandlekar/installed_libraries/robosuite-public/robosuite",
                                        os.path.dirname(robosuite.__file__))           # dataset author's absolute asset paths
    env.reset_from_xml_string(xml)
    states, acts = np.asarray(d["states"]), np.asarray(d["actions"])
    rng = np.random.default_rng([20261021, di]); rows = []
    grip = env.robots[0].gripper
    ramp = [np.zeros(2)]                                                     # the gripper's hidden ramp state (robosuite
    for a in acts[:, -1]:                                                    # PandaGripper.format_action), rebuilt from the
        ramp.append(np.clip(ramp[-1] + np.array([-1.0, 1.0]) * grip.speed * np.sign(a), -1.0, 1.0))   # demo's commands
    for t in range(0, len(acts) - H, H):
        ends, succ = [], []
        for k in range(N):
            env.sim.set_state_from_flattened(states[t]); env.sim.data.qacc_warmstart[:] = 0; env.sim.forward()   # same warm start for every replay
            grip.current_action = ramp[t].copy()
            a = acts[t:t + H].astype(np.float64).copy()
            if k > 0:
                nz = rng.normal(size=(H + 2, 6)) * np.r_[[0.05] * 3, [0.05 / 3] * 3] * scale
                a[:, :6] = np.clip(a[:, :6] + (nz[:-2] + nz[1:-1] + nz[2:]) / 3, -1, 1)
            hold = a[-1].copy(); hold[:6] = 0
            for j in range(H + HOLD):
                env.step(a[j] if j < H else hold)
            ends.append(cube_pose(env)); succ.append(bool(env._check_success()))
        ex = ends[0]
        bad = [any(ex[c][0][2] - e[c][0][2] > 0.02 or e[c][1] - ex[c][1] > 45 for c in range(3)) for e in ends[1:]]
        good = [all(np.linalg.norm(e[c][0] - ex[c][0]) < 0.01 for c in range(3)) for e in ends[1:]]
        nb, ng = int(sum(bad)), int(sum(good))
        rows.append({"demo": di, "t": t, "T": len(acts), "n_bad": nb, "n_good": ng, "fork": nb >= 2 and ng >= 2,
                     "succ_exact": succ[0], "n_succ_noisy": int(sum(succ[1:]))})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--demos", type=int, default=32); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        rows = [r for rs in pool.map(run_demo, [(i, a.scale) for i in range(a.demos)]) for r in rs]
    fork = sum(r["fork"] for r in rows); anyb = sum(r["n_bad"] > 0 for r in rows)
    res = {"env": "StackThree_D1", "noise_scale": a.scale, "demos": a.demos, "points": len(rows), "forks": fork, "points_any_bad": anyb,
           "all_or_nearly_all_bad": sum(r["n_bad"] >= 31 for r in rows),
           "success_splits": sum(2 <= r["n_succ_noisy"] <= 30 for r in rows),
           "pass": bool(fork >= 15 and fork >= 0.25 * max(anyb, 1) and fork >= 0.05 * len(rows)), "seconds": round(time.time() - t0, 1)}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = "" if a.scale == 1 else f"_x{a.scale:g}"
    (out / f"stack_three{tag}_rows.json").write_text(json.dumps(rows)); (out / f"stack_three{tag}_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
