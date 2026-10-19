"""L2 unlabelled world-model data on SafeLIBERO DEVELOPMENT initial states (0-24; 25-49 are the untouched test set).

pi0.5 runs the episode as in policy_eval.py. At up to 4 random decision points per episode the simulator state is
saved and 8 futures are branched from it, each = the policy's full 10-step chunk (f0 as-is, f1-f2 with execution
noise, f3-f7 a uniformly random G transformation plus execution noise) followed by 10 hold steps (zero deltas, last
gripper command) to see what settles. The nominal trajectory is kept too. Frames: agentview, 392 px, every 2 env steps
(one world-model step = 2 env steps = 0.1 s), stored as JPEG q95 for encoding later in the Jenga env (DINOv2 + PCA).

No labels are used. Obstacle and object poses are stored under keys prefixed "evalonly_" for the L3 offline check
only; the training code must never read them.

usage (client env, policy server on :8000):
  python libero/collect_l2.py --suite safelibero_spatial --level I --tasks 0 1 2 3 --episodes 0 1 ... --out DIR
"""
import argparse
import collections
import io
import json
import pathlib
import time
import zlib

import numpy as np
from PIL import Image

import g_transform as G
from policy_eval import DUMMY, MAX_STEPS, quat2axisangle

N_BRANCH, N_FUT, HOLD, FRAME_EVERY, RES = 4, 8, 10, 2, 392


def jpg(img):
    b = io.BytesIO(); Image.fromarray(img).resize((RES, RES), Image.BILINEAR).save(b, "JPEG", quality=95); return b.getvalue()


def frame(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def poses(env, obs, names):
    return np.array([np.r_[obs[f"{n}_pos"], obs[f"{n}_quat"]] for n in names], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_spatial"); ap.add_argument("--level", default="I")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--episodes", type=int, nargs="+", default=list(range(25)))
    ap.add_argument("--out", default="results/libero/l2_data"); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    assert max(a.episodes) <= 24, "development states only (0-24)"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as wcp

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[a.suite](safety_level=a.level)
    client = wcp.WebsocketClientPolicy("0.0.0.0", a.port)
    for tid in a.tasks:
        task = suite.get_task(tid); inits = suite.get_task_init_states(tid)
        env = OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                                 camera_heights=1024, camera_widths=1024)
        env.seed(a.seed)
        objs = [n.replace("_joint0", "") for n in env.sim.model.joint_names if n.endswith("_joint0")]
        for ep in a.episodes:
            f = out / f"{a.suite}_{a.level}_t{tid}_e{ep:02d}.npz"
            if f.exists():
                continue
            t0 = time.time(); rng = np.random.default_rng([20261019, zlib.crc32(f'{a.suite}_{a.level}'.encode()), tid, ep])
            env.reset(); obs = env.set_init_state(inits[ep])
            for _ in range(20):
                obs, _, _, _ = env.step(DUMMY)
            max_steps = MAX_STEPS[a.suite]
            branch_at = set(rng.choice(np.arange(1, (max_steps // 5) * 2 // 3), N_BRANCH, replace=False).tolist())
            plan, t, done, dec = collections.deque(), 0, False, 0
            nom = {"frames": [], "actions": [], "eef": [], "evalonly_poses": []}
            branches = []
            while t < max_steps and not done:
                img = frame(obs)
                if t % FRAME_EVERY == 0:
                    nom["frames"].append(jpg(img)); nom["eef"].append(np.r_[obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]])
                    nom["evalonly_poses"].append(poses(env, obs, objs))
                if not plan:
                    wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    el = {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
                          "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)),
                          "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                          "prompt": str(task.language)}
                    chunk = np.asarray(client.infer(el)["actions"], np.float64)
                    if dec in branch_at:
                        branches.append(branch(env, obs, chunk, objs, rng, t))
                        env.sim.set_state_from_flattened(branches[-1].pop("state")); env.sim.forward()
                        grip(env, branches[-1].pop("grip"))
                    plan.extend(chunk[:5]); dec += 1
                act = np.asarray(plan.popleft()); nom["actions"].append(act)
                obs, _, done, _ = env.step(act.tolist()); t += 1
            np.savez(f, nominal_frames=np.array(nom["frames"], dtype=object), nominal_actions=np.array(nom["actions"], np.float32),
                     nominal_eef=np.array(nom["eef"], np.float32), evalonly_nominal_poses=np.array(nom["evalonly_poses"]),
                     branches=np.array(branches, dtype=object), objects=np.array(objs), success=done, steps=t,
                     language=task.language, gripper_state_restored=True, allow_pickle=True)
            print(json.dumps({"file": f.name, "steps": t, "success": bool(done), "branches": len(branches), "seconds": round(time.time() - t0, 1)}), flush=True)
        env.close()


def grip(env, value=None):
    """State the flattened simulator state does not hold: robosuite's ramped gripper command (current_action, +-0.01
    per step) and MuJoCo's solver warm start. Saved / restored with the state (fix 2026-10-19: files before it lack
    gripper_state_restored; their f1-f7 and post-branch nominal steps start from a slightly different hidden state).
    Also robosuite's step counter and done flag (branch steps must not count toward the episode horizon)."""
    g, e = env.env.robots[0].gripper, env.env
    if value is None:
        return np.array(g.current_action, copy=True), env.sim.data.qacc_warmstart.copy(), e.timestep, e.done
    g.current_action = np.array(value[0], copy=True); env.sim.data.qacc_warmstart[:] = value[1]; e.timestep, e.done = value[2], value[3]


def branch(env, obs0, chunk, objs, rng, t):
    """8 futures from the current state; returns a dict (state is restored by the caller)."""
    st = env.sim.get_state().flatten().copy(); g0 = grip(env)
    T = len(chunk); fut = []; ign = env.env.ignore_done; env.env.ignore_done = True     # futures never end the episode
    for j in range(N_FUT):
        theta = G.IDENTITY.copy() if j < 3 else G.LO + (G.HI - G.LO) * rng.random(4)
        acts = G.transform(chunk, theta) + (G.exec_noise(rng, T) if j >= 1 else 0)
        acts = np.vstack([acts, np.tile(np.r_[np.zeros(6), chunk[-1, 6]], (HOLD, 1))])
        acts[:, :6] = np.clip(acts[:, :6], -1, 1)
        env.sim.set_state_from_flattened(st); env.sim.forward(); grip(env, g0)
        o, frames, ps = obs0, [jpg(frame(obs0))], [poses(env, obs0, objs)]
        for k, a_ in enumerate(acts):
            o, _, _, _ = env.step(a_.tolist())
            if (k + 1) % FRAME_EVERY == 0:
                frames.append(jpg(frame(o))); ps.append(poses(env, o, objs))
        fut.append({"theta": theta.astype(np.float32), "actions": acts.astype(np.float32), "frames": frames, "evalonly_poses": np.array(ps)})
    env.env.ignore_done = ign
    return {"t": t, "chunk": chunk.astype(np.float32), "futures": fut, "state": st, "grip": g0,
            "eef": np.r_[obs0["robot0_eef_pos"], obs0["robot0_eef_quat"], obs0["robot0_gripper_qpos"]].astype(np.float32)}


if __name__ == "__main__":
    main()
