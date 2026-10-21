"""SafeManip world-model data (docs/BENCHMARK_DD.md, check 2b-WM spec): pi0.5 episodes with branched noisy futures at
every 3rd decision, exactly the 2b/2c branching (eval/safemanip_fork_check.py: same restore, noise, labels), saving what
a world model needs. Two views: the fixed monitor camera (robot0_agentview_center, moved by place_monitor_cam) and pi0.5's
wrist camera (robot0_eye_in_hand), 256 x 256, with robot-pixel masks.

  --mode train : exact + N_TRAIN noisy replays per point; frames + masks + proprioception at every model step
                 (2 control steps) over the chunk (20) + hold (10) -> 16 frames per replay; per-step actions
  --mode eval  : exact + 32 noisy replays per point; start frame + end frames (both views, masks); the chunk; per-replay
                 outcomes (slip / unstable release / knocked). Held out: never used for training or tuning.

One file per decision point: <out>/<task>_s<seed>_e<ep>_t<t>.npz, plus <out>/<task>.json (rows without arrays).
Runs in the SafeManip openpi venv with the pi0.5 server on --port (as safemanip_fork_check.py).
"""
import argparse
import collections
import json
import pathlib
import time

import numpy as np

import safemanip_fork_check as fc

H, HOLD, EVERY_DEC, MSTEP = fc.H, fc.HOLD, fc.EVERY_DEC, 2
N_TRAIN, N_EVAL = 4, 32
WM_VIEWS = ("robot0_agentview_center", "robot0_eye_in_hand")       # monitor camera, wrist


def proprio(rs):
    """End-effector position (3), rotation matrix (9), gripper joint positions (2)."""
    D = rs.sim.data; r = rs.robots[0]
    sid = r.eef_site_id["right"]
    g = [D.qpos[rs.sim.model.jnt_qposadr[rs.sim.model.joint_name2id(j)]] for j in r.gripper["right"].joints[:2]]
    return np.r_[D.site_xpos[sid], D.site_xmat[sid], g].astype(np.float32)


def render(rs, rg):
    fv, mv = [], []
    for cn in WM_VIEWS:
        fv.append(rs.sim.render(camera_name=cn, width=256, height=256)[::-1])
        sg = rs.sim.render(camera_name=cn, width=256, height=256, segmentation=True)[::-1]
        gid = sg[..., 1] if sg.ndim == 3 else sg
        mv.append(np.where((gid >= 0) & (gid < len(rg)), rg[np.clip(gid, 0, len(rg) - 1)], False))
    return np.stack(fv).astype(np.uint8), np.stack(mv)


def branch(gw, rs, chunk, convert_action, rng, mode):
    fc.cameras(rs, False)
    s = fc.snapshot(rs); rg = fc.robot_geoms(rs)
    base = np.asarray(chunk[:H], np.float64)
    hold = base[-1].copy(); hold[:6] = 0; hold[7:11] = 0
    E = lambda acts: [fc.to_env_action(gw, x, convert_action) for x in acts]
    n_fut = 1 + (N_TRAIN if mode == "train" else N_EVAL)
    res, F, M, P, A = [], [], [], [], []
    f0, m0 = render(rs, rg); p0 = proprio(rs)
    for k in range(n_fut):
        fc.restore(rs, s); a = base.copy()
        if k > 0:
            n = rng.normal(size=(len(a) + 2, 6)) * np.r_[[0.05] * 3, [0.05 / 3] * 3]
            a[:, :6] = np.clip(a[:, :6] + (n[:-2] + n[1:-1] + n[2:]) / 3, -1, 1)
        full = np.vstack([a, np.tile(hold, (HOLD, 1))])
        traj, cmds = [(fc.obj_state(rs), fc.grasped(rs))], [float(full[0, 6])]
        fr, mk, pp = [f0], [m0], [p0]
        for i, act in enumerate(E(full)):
            rs.step(act)
            traj.append((fc.obj_state(rs), fc.grasped(rs))); cmds.append(float(full[i, 6]))
            if mode == "train" and (i + 1) % MSTEP == 0:
                f, m = render(rs, rg); fr.append(f); mk.append(m); pp.append(proprio(rs))
        res.append(fc.outcome(rs, traj, cmds, traj[-1][0]))
        if mode == "eval":
            f, m = render(rs, rg); fr.append(f); mk.append(m); pp.append(proprio(rs))
        F.append(np.stack(fr)); M.append(np.stack(mk)); P.append(np.stack(pp)); A.append(full.astype(np.float32))
    fc.restore(rs, s); fc.cameras(rs, True)
    bad = [any(r) for r in res]
    row = {"bad_exact": bad[0], "n_bad": int(sum(bad[1:])), "bad_replays": [bool(b) for b in bad],
           "n_slip": int(sum(r[0] for r in res[1:])), "n_unstable": int(sum(r[1] for r in res[1:])),
           "n_knocked": int(sum(r[2] for r in res[1:])), "outcomes": [[bool(x) for x in r] for r in res]}
    arrays = {"frames": np.stack(F), "masks": np.packbits(np.stack(M), axis=-1), "proprio": np.stack(P),
              "actions": np.stack(A), "chunk": np.asarray(chunk, np.float32)}
    return row, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--port", type=int, default=8100); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=("train", "eval"), required=True)
    ap.add_argument("--max-points", type=int, default=0, help="stop an episode after this many branch points (smoke tests)")
    a = ap.parse_args()
    import gymnasium as gym
    import robocasa  # noqa: F401
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as wcp
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    client = wcp.WebsocketClientPolicy("127.0.0.1", a.port)
    rng = np.random.default_rng(20261021 + a.seed)
    for task in a.tasks:
        f = out / f"{task}.json"
        if f.exists():
            continue
        env = gym.make(f"robocasa/{task}", split="pretrain", seed=a.seed); rs = env.unwrapped.env
        horizon = get_task_horizon(task); rows, eps = [], []
        for ep in range(a.episodes):
            t0 = time.time(); obs, info = env.reset(); lang = obs["annotation.human.task_description"]
            fc.place_monitor_cam(rs)
            plan, t, dec, done, npts = collections.deque(), 0, 0, False, 0
            while t < horizon and not done and not (a.max_points and npts >= a.max_points):
                if not plan:
                    im = lambda k: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[k]), 224, 224))
                    state = np.concatenate((obs["state.end_effector_position_relative"], obs["state.end_effector_rotation_relative"],
                                            obs["state.base_position"], obs["state.base_rotation"], obs["state.gripper_qpos"]))
                    chunk = np.asarray(client.infer({"observation/image": im("video.robot0_agentview_left"),
                                                     "observation/wrist_image": im("video.robot0_eye_in_hand"),
                                                     "observation/right_image": im("video.robot0_agentview_right"),
                                                     "observation/state": state, "prompt": lang})["actions"])
                    if dec % EVERY_DEC == 0 and len(chunk) >= 5:
                        row, arr = branch(env.unwrapped, rs, chunk, convert_action, rng, a.mode)
                        row = {"task": task, "episode": ep, "t": t, **row}; rows.append(row); npts += 1
                        np.savez_compressed(out / f"{task}_s{a.seed}_e{ep}_t{t}.npz", **arr)
                    plan.extend(chunk[:5]); dec += 1
                obs, reward, term, trunc, info = env.step(convert_action(plan.popleft()))
                done = bool(info.get("success", False)); t += 1
            eps.append({"episode": ep, "success": done, "steps": t, "points": npts, "seconds": round(time.time() - t0, 1)})
            print(json.dumps({"task": task, **eps[-1]}), flush=True)
        f.write_text(json.dumps({"task": task, "mode": a.mode, "episodes": eps, "rows": rows}))
        env.close()


if __name__ == "__main__":
    main()
