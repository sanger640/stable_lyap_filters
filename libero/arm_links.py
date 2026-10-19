"""Robot link positions (proprioception) along each L2 nominal trajectory, by replaying its actions from the initial
state (restore + replay is bit-identical; checked against the stored end-effector positions). Used for the hand/arm
mask of the monitor; no object state.

usage (client env): python libero/arm_links.py [--shard i n] -> results/libero/arm/<name>.npy (T, 9, 3) every 2 env steps, links 0-7 + hand,
shifted so the replayed end effector matches the recorded one
"""
import argparse
import pathlib
import time

import numpy as np

from policy_eval import DUMMY

LINKS = [f"robot0_link{i}" for i in range(8)] + ["robot0_right_hand"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, nargs=2, default=(0, 1)); a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    data, out = pathlib.Path("results/libero/l2_data"), pathlib.Path("results/libero/arm"); out.mkdir(parents=True, exist_ok=True)
    files = sorted(data.glob("*.npz"))[a.shard[0]::a.shard[1]]
    env, key = None, None
    for f in files:
        o = out / (f.stem + ".npy")
        if o.exists() or time.time() - f.stat().st_mtime < 120:              # skip files the collector may still be writing
            continue
        parts = f.stem.split("_"); s, lv, tid, ep = "_".join(parts[:2]), parts[2], int(parts[3][1:]), int(parts[4][1:])
        suite = benchmark.get_benchmark_dict()[s](safety_level=lv)
        if key != (s, lv, tid):
            if env is not None:
                env.close()
            task = suite.get_task(tid)
            env = OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                                     camera_heights=32, camera_widths=32)
            env.seed(7); key = (s, lv, tid)
        d = np.load(f, allow_pickle=True)
        env.reset(); obs = env.set_init_state(suite.get_task_init_states(tid)[ep])
        for _ in range(20):
            obs, _, _, _ = env.step(DUMMY)
        ids = [env.sim.model.body_name2id(n) for n in LINKS]
        rows, err = [], 0.0
        restored = "gripper_state_restored" in d.files
        br = {int(b["t"]): b for b in d["branches"]}
        g = env.env.robots[0].gripper
        for t, act in enumerate(d["nominal_actions"]):
            if t in br:                       # re-run the branch futures exactly as the collector did (hidden gripper state)
                st = env.sim.get_state().flatten().copy(); g0 = np.array(g.current_action, copy=True); w0 = env.sim.data.qacc_warmstart.copy()
                for fu in br[t]["futures"]:
                    env.sim.set_state_from_flattened(st); env.sim.forward()
                    if restored:
                        g.current_action = g0.copy(); env.sim.data.qacc_warmstart[:] = w0
                    for a_ in fu["actions"]:
                        env.step(np.asarray(a_, np.float64).tolist())
                env.sim.set_state_from_flattened(st); env.sim.forward()
                if restored:
                    g.current_action = g0.copy(); env.sim.data.qacc_warmstart[:] = w0
            if t % 2 == 0:
                rows.append(np.vstack([env.sim.data.body_xpos[ids], obs["robot0_eef_pos"][None]]))
                e = float(np.abs(obs["robot0_eef_pos"] - d["nominal_eef"][t // 2][:3]).max())
                if e > 1e-4 and err <= 1e-4: print("first mismatch at", t)
                err = max(err, e)
            obs, _, _, _ = env.step(act.tolist())
        # replay is approximate for files collected before the hidden-state fix: shift the arm so the hand matches the
        # recorded end-effector at each frame (the stored eef is the truth; the arm shape comes from the replay)
        rows = np.array(rows, np.float32); n = min(len(rows), len(d["nominal_eef"]))
        shift = d["nominal_eef"][:n, :3] - rows[:n, -1]
        np.save(o, rows[:n, :-1] + shift[:, None])
        print(f.stem, len(rows), f"max eef mismatch {err * 1000:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
