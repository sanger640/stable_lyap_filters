"""ToolHang RECHECK (docs/BENCHMARK_RECHECK.md): the original check 1 (eval/toolhang_fork_check.py) used damage-only
labels (slip / unstable release / knocked). This rerun keeps the same policy, seeds, decision points, restore, noise and
20-step hold, and adds:
  - a generic "outcome changed" label vs the exact replay's end: BAD = some object (stand, frame, tool) ends > 2 cm or
    > 20 deg away from the exact replay's end; GOOD = every object within 1 cm. Fork = >= 2 BAD and >= 2 GOOD.
  - the task's own success flag at the end of the window (split = 2..30 of 32 noisy replays differ);
  - the realised gripper (eef) deviation from the exact replay at the end of the chunk (noise-level check);
  - --scale 0 = zero-noise control (must give no forks).
No rendering. Runs in ~/wksp/robomimic_bench/.venv.
usage: MUJOCO_GL=egl python eval/toolhang_recheck.py --ckpt CKPT --seeds 1 2 --out DIR [--scale 1]
"""
import argparse
import json
import pathlib
import time

import numpy as np

import safemanip_fork_check as fc

H, HOLD, N, EVERY_DEC, HORIZON = 8, 20, 33, 2, 800


def poses(rs):
    return {n: (p.copy(), t) for n, (p, t) in fc.obj_state(rs).items()}


def branch(rs, chunk, rng, scale):
    s = fc.snapshot(rs)
    base = np.asarray(chunk[:H], np.float64); hold = base[-1].copy(); hold[:6] = 0
    ends, succ, eef = [], [], []
    for k in range(N):
        fc.restore(rs, s); a = base.copy()
        if k > 0:
            n = rng.normal(size=(len(a) + 2, 6)) * np.r_[[0.05] * 3, [0.05 / 3] * 3] * scale
            a[:, :6] = np.clip(a[:, :6] + (n[:-2] + n[1:-1] + n[2:]) / 3, -1, 1)
        for i, act in enumerate(np.vstack([a, np.tile(hold, (HOLD, 1))])):
            rs.step(act)
            if i == H - 1:
                eef.append(rs.sim.data.site_xpos[rs.robots[0].eef_site_id["right"]].copy())
        ends.append(poses(rs)); succ.append(bool(rs._check_success()))
    fc.restore(rs, s)
    ex = ends[0]
    bad = [any(np.linalg.norm(e[n][0] - ex[n][0]) > 0.02 or abs(e[n][1] - ex[n][1]) > 20 for n in ex) for e in ends[1:]]
    good = [all(np.linalg.norm(e[n][0] - ex[n][0]) < 0.01 for n in ex) for e in ends[1:]]
    nb, ng = int(sum(bad)), int(sum(good)); ns = int(sum(succ[1:]))
    return {"n_bad": nb, "n_good": ng, "fork": nb >= 2 and ng >= 2, "succ_exact": succ[0], "n_succ_noisy": ns,
            "succ_split": 2 <= ns <= 30, "eef_dev_mm_median": float(np.median(np.linalg.norm(np.array(eef[1:]) - eef[0], axis=1)) * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    import torch
    import robomimic.utils.file_utils as FU
    policy, ckpt = FU.policy_from_checkpoint(ckpt_path=a.ckpt, device="cuda", verbose=False)
    env, _ = FU.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=False, verbose=False)
    rs = env
    while not hasattr(rs, "sim"):
        rs = rs.env
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = "" if a.scale == 1 else f"_x{a.scale:g}"
    for seed in a.seeds:
        f = out / f"ToolHang_s{seed}{tag}.json"
        if f.exists():
            continue
        np.random.seed(seed); torch.manual_seed(seed)                       # same seeding as the original check
        rng = np.random.default_rng(20261021 + seed)
        t0 = time.time(); policy.start_episode(); obs = env.reset()
        rs.objects = {"stand": rs.stand, "frame": rs.frame, "tool": rs.tool}
        rows, t, dec, done = [], 0, 0, False
        while t < HORIZON and not done:
            new = len(policy.policy.action_queue) == 0
            act = policy(ob=obs)
            if new:
                chunk = np.vstack([act] + [x.detach().cpu().numpy().reshape(-1) for x in policy.policy.action_queue])
                if dec % EVERY_DEC == 0:
                    rows.append({"t": t, **branch(rs, chunk, rng, a.scale)})
                dec += 1
            obs, r, _, _ = env.step(act); t += 1
            done = bool(env.is_success()["task"])
        ep = {"seed": seed, "success": done, "steps": t, "points": len(rows), "seconds": round(time.time() - t0, 1)}
        print(json.dumps(ep), flush=True)
        f.write_text(json.dumps({"episode": ep, "rows": rows}))


if __name__ == "__main__":
    main()
