"""ManiSkill3 videos for the dashboard (docs/BENCHMARK_DD.md, quick fork check), for LiftPegUpright and StackPyramid.
1) policy episodes: the shipped expert demos (LiftPegUpright: PPO; StackPyramid: motion planner) replayed exactly from their
   saved states; left = fixed side camera (for LiftPegUpright the chosen world-model camera), right = an oblique overview.
2) fork clip: at the task's most fork-like moment from eval/maniskill_fork_check.py, 33 parallel copies run the next 8
   policy actions (exact + 32 with ~2.5 mm smoothed noise) + 20 hold steps. LiftPegUpright: one replay that stands
   (< 20 deg) and one that falls flat (> 60 deg). StackPyramid: the two noisy replays whose cubes end farthest from the
   exact replay's (to show what its rare "forks" look like).
Runs in ~/wksp/maniskill_bench/.venv.  usage: python eval/maniskill_video.py TASK OUTDIR
"""
import json
import pathlib
import sys

import cv2
import imageio
import numpy as np

ROOT = pathlib.Path.home() / "wksp/maniskill_bench"
TASKS = {
    "LiftPegUpright-v1": dict(h5="demos/LiftPegUpright-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5", mode="pd_ee_delta_pose",
                              policy="PPO policy", demos=(3, 7), objs=("peg",), sig=[0.025] * 3 + [0.025 / 3] * 3,
                              rows="fc_quick_poses/LiftPegUpright-v1_rows.json",
                              cams={"wm_side": ([0.0, -0.5, 0.15], [0.0, 0.0, 0.05]), "overview": ([0.35, -0.35, 0.3], [0.0, 0.0, 0.05])}),
    "StackPyramid-v1": dict(h5="demos/StackPyramid-v1/motionplanning/trajectory.h5", mode="pd_joint_pos",
                            policy="motion-planner policy", demos=(5, 11), objs=("cubeA", "cubeB", "cubeC"), sig=[0.004] * 7,
                            rows="fc_quick/StackPyramid-v1_rows.json",
                            cams={"side": ([0.0, -0.45, 0.15], [0.0, 0.0, 0.05]), "overview": ([0.35, -0.35, 0.3], [0.0, 0.0, 0.05])}),
}
GREY, GREEN, RED = (60, 60, 70), (40, 150, 60), (190, 50, 40)


def make(task, n, cam):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import sapien_utils
    c = TASKS[task]; e, tg = c["cams"][cam]
    env = gym.make(task, num_envs=n, obs_mode="none", control_mode=c["mode"], render_mode="rgb_array", sim_backend="physx_cuda",
                   human_render_camera_configs=dict(width=384, height=384, pose=sapien_utils.look_at(e, tg), fov=1.0))
    env.reset(seed=0)
    return env


def state(es, t, n):
    import torch
    return {g: {k: torch.as_tensor(np.asarray(es[g][k][t]))[None].float().cuda().expand(n, -1).contiguous() for k in es[g].keys()} for g in es.keys()}


def tilt(q):
    """Angle (deg) of a peg's long axis (local x) from vertical."""
    w, x, y, z = q
    return float(np.degrees(np.arccos(min(1.0, abs(2 * (x * z - w * y))))))


def banner(img, txt, col):
    bar = np.full((34, img.shape[1], 3), col, np.uint8)
    cv2.putText(bar, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main(task, out):
    import h5py
    import torch
    c = TASKS[task]; short = task.split("-")[0]
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(ROOT / c["h5"], "r")
    cams = list(c["cams"])
    envs = {k: make(task, 1, k) for k in cams}
    for di in c["demos"]:                                                    # 1) policy episodes
        tr = f[f"traj_{di}"]; es = tr["env_states"]; T = len(es["actors"][c["objs"][0]]); frames = []
        for t in range(T):
            row = []
            for k, env in envs.items():
                env.unwrapped.set_state_dict(state(es, t, 1)); env.unwrapped.scene._gpu_fetch_all()
                row.append(env.render()[0].cpu().numpy())
            succ = bool(tr["success"][min(t, T - 2)])
            frames.append(banner(np.concatenate(row, 1), f"{short}, {c['policy']}   step {t}/{T - 1}   left: fixed side camera   right: overview"
                                 + ("   SUCCESS" if succ else ""), GREEN if succ else GREY))
        rep = 2 if T < 100 else 1                                            # short LiftPeg episodes at half speed
        frames = [x for x in frames for _ in range(rep)] + [frames[-1]] * 30
        imageio.mimwrite(out / f"{short}_policy_demo{di}.mp4", frames, fps=20, quality=6, macro_block_size=2)
        print("policy video", di, T, flush=True)
    for env in envs.values():
        env.close()
    rows = json.loads((ROOT / c["rows"]).read_text())                         # 2) fork clip
    if task == "LiftPegUpright-v1":
        nfall = lambda r: sum(tilt(q) > 60 for q in np.array(r["final_pose"]["peg"])[1:, 3:])
        cand = sorted(rows, key=lambda r: abs(nfall(r) - 16))
    else:
        cand = sorted(rows, key=lambda r: -r["n_bad"])
    env = make(task, 33, cams[0]); u = env.unwrapped
    rng = np.random.default_rng(7); sig = np.asarray(c["sig"])
    for r in cand:
        di, t = r["demo"], r["t"]
        tr = f[f"traj_{di}"]; es = tr["env_states"]; acts = np.asarray(tr["actions"])
        u.set_state_dict(state(es, t, 33))
        base = np.repeat(acts[t:t + 8][None], 33, 0).astype(np.float64)
        nz = rng.normal(size=(33, 10, len(sig))) * sig; nz = (nz[:, :-2] + nz[:, 1:-1] + nz[:, 2:]) / 3; nz[0] = 0
        base[:, :, :len(sig)] += nz
        if c["mode"] == "pd_ee_delta_pose":
            base[:, :, :6] = np.clip(base[:, :, :6], -1, 1); hold = base[:, -1].copy(); hold[:, :6] = 0
        else:
            hold = base[:, -1].copy()
        vid = [env.render().cpu().numpy()]
        for k in range(28):
            env.step(torch.as_tensor(base[:, k] if k < 8 else hold, device=u.device).float())
            vid.append(env.render().cpu().numpy())
        if task == "LiftPegUpright-v1":
            ang = np.array([tilt(q) for q in u.peg.pose.q.cpu().numpy()])
            ok, bad = np.where(ang[1:] < 20)[0] + 1, np.where(ang[1:] > 60)[0] + 1
            if len(ok) and len(bad):
                i, j = ok[0], bad[0]
                lab = [f"STANDS ({ang[0]:.0f} deg)" if ang[0] < 20 else f"FALLS ({ang[0]:.0f} deg)", f"STANDS ({ang[i]:.0f} deg)", f"FALLS FLAT ({ang[j]:.0f} deg)"]
                cols = [GREY, GREEN, RED]; note = f"standing {int((ang[1:] < 20).sum())}, flat {int((ang[1:] > 60).sum())} of 32"
                break
        else:
            d = sum((getattr(u, o).pose.p - getattr(u, o).pose.p[0:1]).norm(dim=1) for o in c["objs"]).cpu().numpy() * 1000
            order = np.argsort(-d[1:]) + 1; i, j = order[0], order[1]
            lab = ["exact", f"cubes moved {d[i]:.0f} mm vs exact", f"cubes moved {d[j]:.0f} mm vs exact"]
            cols = [GREY, RED if d[i] > 20 else GREY, RED if d[j] > 20 else GREY]
            note = f"replays > 20 mm off: {int((d[1:] > 20).sum())} of 32"
            break
    frames = []
    for k, v in enumerate(vid):
        last = k == len(vid) - 1
        tiles = [banner(v[x], name + (f"   {lb}" if last else ""), cl if last else GREY)
                 for x, name, lb, cl in zip((0, i, j), ("exact policy actions", "noise draw A (~2.5 mm)", "noise draw B (~2.5 mm)"), lab, cols)]
        frames.append(np.concatenate(tiles, 1))
    frames = [x for x in frames for _ in range(3)] + [frames[-1]] * 40
    imageio.mimwrite(out / f"{short}_fork.mp4", frames, fps=20, quality=6, macro_block_size=2)
    print("fork clip", di, t, note, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
