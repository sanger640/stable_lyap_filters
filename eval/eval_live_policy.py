"""
The decisive test for diffusion steering (section 8.5): does guidance actually reduce REAL
topples in MuJoCo, or does it only lower the world model's own predicted divergence score?

Sections 8.2-8.4 only established the latter. That is exactly the failure mode the toy-task
detour surfaced on CartPole (RESUME 7.43): a policy can learn to satisfy a learned model's
predictions while real behaviour does not improve at all. So this runs the actual policy in
the actual simulator, closed loop, and counts real topples.

Design decisions that matter for the result being meaningful:

* PAIRED trials. Trial i uses the same seed for both conditions, so `reset_simulation`'s
  random block perturbation is IDENTICAL for steered and unsteered. Expected effect is
  modest and topple rate is noisy, so an unpaired comparison at feasible N would be mostly
  noise. Paired lets each trial act as its own control.

* TASK SUCCESS IS MEASURED, NOT JUST SAFETY. A policy that freezes and does nothing has a
  perfect topple rate. That degenerate "safe" optimum is not hypothetical -- it is precisely
  what the Pusher2D actor converged to (RESUME 7.43) and it looked fine on the safety metric
  alone. So every trial also records whether the target block was actually lifted, and the
  headline result is only meaningful if steering preserves task success.

* Results are appended to disk per-trial (JSONL). The EGL context throws at interpreter
  teardown on this machine (harmless, post-run), but this project has already lost an hour of
  GPU work once to a crash after compute finished (RESUME 7.29), so nothing is held only in
  memory until the end.
"""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("SIM_HEADLESS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import add_external_paths, EPISODES_DIR, check, enter_sim_dir  # noqa: E402
check()
add_external_paths(dino_wm=True, panda_express=True, diffusion_policy=True)
enter_sim_dir()   # sim.py resolves its MuJoCo XML relative to the panda_express root

import mujoco  # noqa: E402
from sim import SimRobotInterface, SimDualCamera, SIM, TOPPLE_THRESHOLD_DEG  # noqa: E402

from steering import (load_frozen_world_model, dino_preprocess,  # noqa: E402
                      steered_conditional_sample)
from test_steering import load_diffusion_policy, DEVICE  # noqa: E402

WM_HIST, DP_HIST = 3, 2
CONTROL_HZ = 10.0
LIFT_SUCCESS_M = 0.03   # target block raised this far above its start height = picked


def block_z(name):
    bid = mujoco.mj_name2id(SIM.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return float(SIM.data.xpos[bid][2]) if bid != -1 else 0.0


def run_trial(policy, world_model, robot, cams, guidance_scale, seed, max_steps, horizon_exec):
    """One closed-loop episode. Returns a dict of outcome metrics."""
    np.random.seed(seed)          # makes reset_simulation's block noise reproducible/paired
    torch.manual_seed(seed)
    robot.reset()
    time.sleep(0.6)               # let the physics settle after reset

    start_z = block_z("block_middle")
    cam1_hist, cam2_hist, prop_hist, act_hist = deque(maxlen=WM_HIST), deque(maxlen=WM_HIST), \
        deque(maxlen=WM_HIST), deque(maxlen=WM_HIST)

    toppled, topple_step, peak_tilt, peak_lift = False, None, 0.0, 0.0
    step, t_infer_total, n_infer = 0, 0.0, 0

    while step < max_steps:
        if len(cam1_hist) < WM_HIST:
            # priming: one capture per control step, so the history the first inference sees
            # is consecutive (captures during chunk execution below keep it that way)
            c1, c2 = cams.get_frames()
            state = robot.get_state()
            cam1_hist.append(c1); cam2_hist.append(c2); prop_hist.append(state)
            act_hist.append(state.copy() if not act_hist else act_hist[-1].copy())
            time.sleep(1.0 / CONTROL_HZ)
            step += 1
            continue

        obs_dict = {
            "camera_1": torch.from_numpy(np.stack(list(cam1_hist)[-DP_HIST:])).float()
                            .unsqueeze(0).to(DEVICE) / 255.0,
            "camera_2": torch.from_numpy(np.stack(list(cam2_hist)[-DP_HIST:])).float()
                            .unsqueeze(0).to(DEVICE) / 255.0,
            "agent_pos": torch.from_numpy(np.stack(list(prop_hist)[-DP_HIST:])).float()
                            .unsqueeze(0).to(DEVICE),
        }
        visual_dino = dino_preprocess(
            torch.from_numpy(np.stack(cam2_hist)).unsqueeze(0), DEVICE)
        proprio_dino = torch.from_numpy(np.stack(prop_hist)).float().unsqueeze(0).to(DEVICE)

        t0 = time.time()
        nobs = policy.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        with torch.no_grad():
            global_cond = policy.obs_encoder(this_nobs).reshape(B, -1)
        cond_data = torch.zeros((B, policy.horizon, policy.action_dim), device=DEVICE)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        gen = torch.Generator(device=DEVICE).manual_seed(seed * 1000 + step)
        with torch.enable_grad():
            nsample, _ = steered_conditional_sample(
                policy, cond_data, cond_mask, world_model, visual_dino, proprio_dino,
                guidance_scale=guidance_scale, global_cond=global_cond, generator=gen)
        action_pred = policy.normalizer["action"].unnormalize(nsample[..., :policy.action_dim])
        t_infer_total += time.time() - t0
        n_infer += 1

        chunk = action_pred[0, DP_HIST - 1: DP_HIST - 1 + horizon_exec].detach().cpu().numpy()
        for a in chunk:
            # capture observation BEFORE commanding, every control step -- an earlier version
            # only refreshed the observation deques once per CHUNK, so both the policy and the
            # world model received history frames `horizon_exec` steps apart instead of
            # consecutive ones, badly out of distribution for models trained on 10 Hz
            # consecutive history (and misaligned against act_hist, which was updated per step)
            c1_s, c2_s = cams.get_frames()
            cam1_hist.append(c1_s); cam2_hist.append(c2_s)
            prop_hist.append(robot.get_state())

            robot.execute(a.astype(np.float64))
            act_hist.append(a.astype(np.float32))
            time.sleep(1.0 / CONTROL_HZ)
            step += 1

            failed, _, tilt = SIM.check_failure(TOPPLE_THRESHOLD_DEG)
            peak_tilt = max(peak_tilt, tilt)
            peak_lift = max(peak_lift, block_z("block_middle") - start_z)
            if failed and not toppled:
                toppled, topple_step = True, step
            if step >= max_steps:
                break

    return {
        "seed": seed, "guidance_scale": guidance_scale,
        "toppled": bool(toppled), "topple_step": topple_step,
        "peak_tilt_deg": round(peak_tilt, 3),
        "peak_lift_m": round(peak_lift, 4),
        "picked": bool(peak_lift > LIFT_SUCCESS_M),
        "steps": step,
        "mean_inference_s": round(t_infer_total / max(n_infer, 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--horizon-exec", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default="results/steering_sim_eval.jsonl")
    args = ap.parse_args()

    if not Path(args.out).is_absolute():
        args.out = str(Path(__file__).resolve().parent.parent / args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    world_model = load_frozen_world_model(DEVICE)
    policy = load_diffusion_policy()
    robot = SimRobotInterface()
    cams = SimDualCamera()

    fout = open(args.out, "a")
    rows = []
    for i in range(args.n_trials):
        seed = args.seed0 + i
        for scale in (0.0, args.guidance_scale):     # paired: same seed, same initial scene
            r = run_trial(policy, world_model, robot, cams, scale, seed,
                          args.max_steps, args.horizon_exec)
            rows.append(r)
            fout.write(json.dumps(r) + "\n"); fout.flush()   # never hold results only in RAM
            tag = "UNSTEER" if scale == 0 else f"STEER s={scale:g}"
            print(f"[trial {i+1}/{args.n_trials} seed={seed}] {tag:<12} "
                  f"toppled={r['toppled']!s:<5} picked={r['picked']!s:<5} "
                  f"peak_tilt={r['peak_tilt_deg']:6.2f}  lift={r['peak_lift_m']:.3f}  "
                  f"infer={r['mean_inference_s']:.2f}s", flush=True)
    fout.close()

    def agg(scale):
        s = [r for r in rows if r["guidance_scale"] == scale]
        n = max(len(s), 1)
        return (sum(r["toppled"] for r in s) / n, sum(r["picked"] for r in s) / n,
                float(np.mean([r["peak_tilt_deg"] for r in s])) if s else float("nan"), len(s))

    u_top, u_pick, u_tilt, n_u = agg(0.0)
    s_top, s_pick, s_tilt, n_s = agg(args.guidance_scale)
    paired = {}
    for r in rows:
        paired.setdefault(r["seed"], {})[r["guidance_scale"]] = r
    both = [v for v in paired.values() if len(v) == 2]
    helped = sum(1 for v in both if v[0.0]["toppled"] and not v[args.guidance_scale]["toppled"])
    hurt = sum(1 for v in both if not v[0.0]["toppled"] and v[args.guidance_scale]["toppled"])

    print(f"\n{'='*72}\nRESULTS -- live MuJoCo, {len(both)} paired trials\n{'='*72}")
    print(f"  UNSTEERED         topple {u_top:.1%}  picked {u_pick:.1%}  mean peak tilt {u_tilt:.2f} deg  (n={n_u})")
    print(f"  STEERED s={args.guidance_scale:<7g} topple {s_top:.1%}  picked {s_pick:.1%}  mean peak tilt {s_tilt:.2f} deg  (n={n_s})")
    print(f"  paired: steering PREVENTED {helped} topple(s), CAUSED {hurt}")
    print("  (task success 'picked' must hold up -- a frozen policy would score 0% topples)")
    json.dump({"unsteered": {"topple_rate": u_top, "pick_rate": u_pick, "mean_peak_tilt": u_tilt, "n": n_u},
               "steered": {"topple_rate": s_top, "pick_rate": s_pick, "mean_peak_tilt": s_tilt, "n": n_s},
               "paired_prevented": helped, "paired_caused": hurt, "n_paired": len(both)},
              open(str(Path(args.out).with_suffix(".summary.json")), "w"), indent=1)


if __name__ == "__main__":
    main()
