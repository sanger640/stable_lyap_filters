"""LiftPegUpright videos for the dashboard (docs/BENCHMARK_DD.md, ManiSkill3 quick fork check).
1) policy episodes: the shipped PPO demos replayed exactly from their saved states; left = the chosen world-model camera
   (fixed side camera, peg in profile), right = an oblique overview.
2) fork clip: from a fork moment of eval/maniskill_fork_check.py, 33 parallel copies run the next 8 policy actions
   (exact + 32 with ~2.5 mm smoothed noise) + 20 hold steps; shows one replay that stands (< 20 deg) and one that falls flat (> 60 deg),
   both from the world-model camera, plus the exact replay.
Runs in ~/wksp/maniskill_bench/.venv.  usage: python eval/maniskill_video.py OUTDIR
"""
import json
import pathlib
import sys

import cv2
import imageio
import numpy as np

ROOT = pathlib.Path.home() / "wksp/maniskill_bench"
H5 = ROOT / "demos/LiftPegUpright-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
CAMS = {"wm_side": ([0.0, -0.5, 0.15], [0.0, 0.0, 0.05]), "overview": ([0.35, -0.35, 0.3], [0.0, 0.0, 0.05])}
SIG = np.array([0.025] * 3 + [0.025 / 3] * 3)


def make(n, cam):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import sapien_utils
    e, tg = CAMS[cam]
    env = gym.make("LiftPegUpright-v1", num_envs=n, obs_mode="none", control_mode="pd_ee_delta_pose", render_mode="rgb_array",
                   sim_backend="physx_cuda", human_render_camera_configs=dict(width=384, height=384, pose=sapien_utils.look_at(e, tg), fov=1.0))
    env.reset(seed=0)
    return env


def state(es, t, n):
    import torch
    return {g: {k: torch.as_tensor(np.asarray(es[g][k][t]))[None].float().cuda().expand(n, -1).contiguous() for k in es[g].keys()} for g in es.keys()}


def tilt(q):
    """Angle (deg) of the peg's long axis (local x) from vertical."""
    w, x, y, z = q
    return float(np.degrees(np.arccos(min(1.0, abs(2 * (x * z - w * y))))))


def banner(img, txt, col):
    bar = np.full((34, img.shape[1], 3), col, np.uint8)
    cv2.putText(bar, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main(out):
    import h5py
    import torch
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(H5, "r")
    envs = {c: make(1, c) for c in CAMS}
    for di in (3, 7):                                                        # 1) policy episodes
        es = f[f"traj_{di}"]["env_states"]; T = len(es["actors"]["peg"]); frames = []
        for t in range(T):
            row = []
            for c, env in envs.items():
                env.unwrapped.set_state_dict(state(es, t, 1)); env.unwrapped.scene._gpu_fetch_all()
                row.append(env.render()[0].cpu().numpy())
            succ = bool(f[f"traj_{di}"]["success"][min(t, T - 2)])
            frames.append(banner(np.concatenate(row, 1), f"LiftPegUpright, PPO policy   step {t}/{T - 1}   left: world-model camera   right: overview"
                                 + ("   UPRIGHT" if succ else ""), (40, 150, 60) if succ else (60, 60, 70)))
        frames = [x for x in frames for _ in range(2)] + [frames[-1]] * 30   # 0.5x speed (policy runs at 20 Hz, sim shown at 10)
        imageio.mimwrite(out / f"LiftPeg_policy_demo{di}.mp4", frames, fps=20, quality=6, macro_block_size=2)
        print("policy video", di, T, flush=True)
    for env in envs.values():
        env.close()
    rows = json.loads((ROOT / "fc_quick_poses/LiftPegUpright-v1_rows.json").read_text())   # 2) fork clip: a two-mode moment
    nfall = lambda r: sum(tilt(q) > 60 for q in np.array(r["final_pose"]["peg"])[1:, 3:])
    env = make(33, "wm_side"); u = env.unwrapped
    rng = np.random.default_rng(7)
    for r in sorted(rows, key=lambda r: abs(nfall(r) - 16)):
        di, t = r["demo"], r["t"]
        tr = f[f"traj_{di}"]; es = tr["env_states"]; acts = np.asarray(tr["actions"])
        u.set_state_dict(state(es, t, 33))
        base = np.repeat(acts[t:t + 8][None], 33, 0).astype(np.float64)
        nz = rng.normal(size=(33, 10, 6)) * SIG; nz = (nz[:, :-2] + nz[:, 1:-1] + nz[:, 2:]) / 3; nz[0] = 0
        base[:, :, :6] = np.clip(base[:, :, :6] + nz, -1, 1); hold = base[:, -1].copy(); hold[:, :6] = 0
        vid = [env.render().cpu().numpy()]
        for k in range(28):
            env.step(torch.as_tensor(base[:, k] if k < 8 else hold, device=u.device).float())
            vid.append(env.render().cpu().numpy())
        ang = np.array([tilt(q) for q in u.peg.pose.q.cpu().numpy()]); s = ang < 20
        ok, bad = np.where(ang[1:] < 20)[0] + 1, np.where(ang[1:] > 60)[0] + 1
        if len(ok) and len(bad):
            break
    i, j = ok[0], bad[0]
    frames = []
    for k, v in enumerate(vid):
        last = k == len(vid) - 1
        a = banner(v[0], "exact policy actions" + (f"   {'STANDS' if s[0] else 'FALLS'} ({ang[0]:.0f} deg)" if last else ""), (60, 60, 70))
        b = banner(v[i], "noise draw A (~2.5 mm)" + (f"   STANDS ({ang[i]:.0f} deg)" if last else ""), (40, 150, 60) if last else (60, 60, 70))
        c = banner(v[j], "noise draw B (~2.5 mm)" + (f"   FALLS FLAT ({ang[j]:.0f} deg)" if last else ""), (190, 50, 40) if last else (60, 60, 70))
        frames.append(np.concatenate([a, b, c], 1))
    frames = [x for x in frames for _ in range(3)] + [frames[-1]] * 40
    imageio.mimwrite(out / "LiftPeg_fork.mp4", frames, fps=20, quality=6, macro_block_size=2)
    print("fork clip", di, t, "standing", int((ang[1:] < 20).sum()), "flat", int((ang[1:] > 60).sum()), "of 32", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
