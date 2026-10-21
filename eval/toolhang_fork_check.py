"""robomimic ToolHang, checks 1 + 2 (docs/BENCHMARK_DD.md, spec fixed before running): do failures FORK under small
execution noise when our robomimic Diffusion Policy runs ToolHang, and does the robot-masked split test see them?

Same branching, restore, noise and outcome labels as SafeManip (eval/safemanip_fork_check.py, reused): at every 2nd
decision (the policy's chunk of 8 actions), 33 replays of the chunk + 20 hold steps: exact + 32 with smoothed Gaussian arm
noise; outcomes from simulator state (slip / unstable release / knocked); end frames of every replay in 3 views
(agentview = gate, sideview, wrist) with robot-pixel masks. Frames saved for every decision point.

Runs in ~/wksp/robomimic_bench/.venv (robosuite 1.5.1, robomimic 0.5 from source).
usage: MUJOCO_GL=egl python eval/toolhang_fork_check.py --ckpt CKPT --seeds 1 2 --out DIR
"""
import argparse
import json
import pathlib
import time

import numpy as np

import safemanip_fork_check as fc

fc.H, fc.HOLD, fc.N_FUT = 8, 20, 33
fc.VIEWS = ("agentview", "sideview", "robot0_eye_in_hand")
fc.to_env_action = lambda gw, a, convert_action: np.asarray(a, np.float64)       # policy actions are env actions
EVERY_DEC, HORIZON = 2, 800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--no-branch", action="store_true", help="success rate only")
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
        f = out / f"ToolHang_s{seed}.json"
        if f.exists():
            continue
        np.random.seed(seed); torch.manual_seed(seed)
        rng = np.random.default_rng(20261021 + seed)
        t0 = time.time(); policy.start_episode(); obs = env.reset()
        rs.objects = {"stand": rs.stand, "frame": rs.frame, "tool": rs.tool}
        rows, t, dec, done, npts = [], 0, 0, False, 0
        q = policy.policy.action_queue
        while t < HORIZON and not done:
            new = len(policy.policy.action_queue) == 0
            act = policy(ob=obs)
            if new:
                chunk = np.vstack([act] + [x.detach().cpu().numpy().reshape(-1) for x in policy.policy.action_queue])
                if not a.no_branch and dec % EVERY_DEC == 0 and not (a.max_points and npts >= a.max_points):
                    row = fc.branch(None, rs, chunk, None, rng, "ToolHang", 0, t, render=True, masked=True)
                    fr, mk = row.pop("frames"), row.pop("masks")
                    np.save(out / f"ToolHang_s{seed}_e0_t{t}.npy", fr)
                    np.save(out / f"ToolHang_s{seed}_e0_t{t}_mask.npy", np.packbits(mk, axis=-1))
                    rows.append(row); npts += 1
                dec += 1
            obs, r, _, _ = env.step(act); t += 1
            done = bool(env.is_success()["task"])
        ep = {"seed": seed, "success": done, "steps": t, "points": npts, "seconds": round(time.time() - t0, 1),
              "exact_ok": float(np.mean([x["exact_ok"] for x in rows])) if rows else None}
        print(json.dumps(ep), flush=True)
        f.write_text(json.dumps({"task": "ToolHang", "episodes": [ep], "rows": rows}))


if __name__ == "__main__":
    main()
