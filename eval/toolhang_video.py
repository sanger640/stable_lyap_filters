"""ToolHang policy videos for the lab dashboard: our robomimic Diffusion Policy (epoch-600 checkpoint) running alone, same
seeding as eval/toolhang_fork_check.py (so outcomes match the 50-episode re-measure). Agentview (the fork check's gate
view) and frontview side by side (in ToolHang robosuite puts sideview 5 cm below agentview, same angle), real time (20 Hz control), banner with the outcome from the start and the step count.
Runs in ~/wksp/robomimic_bench/.venv.
usage: MUJOCO_GL=egl python eval/toolhang_video.py --ckpt CKPT --seeds 103 101 --out DIR
"""
import argparse
import pathlib

import cv2
import imageio
import numpy as np

HORIZON, RES = 800, 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import robomimic.utils.file_utils as FU
    policy, ckpt = FU.policy_from_checkpoint(ckpt_path=a.ckpt, device="cuda", verbose=False)
    env, _ = FU.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=True, verbose=False)
    rs = env
    while not hasattr(rs, "sim"):
        rs = rs.env
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        policy.start_episode(); obs = env.reset()
        frames, t, done = [], 0, False
        while t < HORIZON and not done:
            act = policy(ob=obs)
            obs, _, _, _ = env.step(act); t += 1
            done = bool(env.is_success()["task"])
            img = np.concatenate([rs.sim.render(camera_name=c, width=RES, height=RES)[::-1] for c in ("agentview", "frontview")], 1)
            frames.append(img)
        tag = "success" if done else "fail"
        col = (40, 150, 60) if done else (190, 50, 40)
        for i, f in enumerate(frames):
            bar = np.full((36, f.shape[1], 3), 30, np.uint8)
            bar[:] = col                                                  # outcome shown from the start, as the SafeManip videos
            txt = f"ToolHang, Diffusion Policy alone   outcome: {tag.upper()}   step {min(i + 1, t)}/{HORIZON}"
            cv2.putText(bar, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            frames[i] = np.concatenate([bar, f], 0)
        frames += [frames[-1]] * 40                                       # hold the final frame 2 s
        p = out / f"ToolHang_s{seed}_{tag}.mp4"
        imageio.mimwrite(p, frames, fps=20, quality=6, macro_block_size=8)
        print(p, t, tag, flush=True)


if __name__ == "__main__":
    main()
