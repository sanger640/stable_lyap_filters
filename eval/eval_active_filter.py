"""
The decisive test, run on data where real topples actually happen (section 8.6).

Why not the live-policy version (eval_steering_sim.py): the trained Jenga diffusion policy
never completes the pick (gripper RMSE 0.48-0.67 on a +-1 signal, section 8.5), so it never
generates topple risk -- measured base topple rate 0/16 with peak tilt ~10 deg against a 45
deg threshold. Comparing steered vs unsteered topple rate there is vacuous: 0% vs 0%.

This instead uses the setting that DOES produce real failures: replay a clean expert teleop
trajectory through MuJoCo with Gaussian position noise injected per waypoint, exactly as
replay_noisy.py generates the labelled noisy dataset (~33% topple rate at pos_std=0.002 per
RESUME 5d). The "unsafe action sequence" is then real and physical, not hypothetical.

Two paired conditions on identical noise realisations and identical initial scenes:
  CONTROL  -- execute the noisy plan as recorded
  FILTERED -- before executing each chunk of H waypoints, run K steps of gradient descent on
              that chunk's positions to minimise the frozen world model's predicted
              divergence, then execute the corrected chunk

This is CLAUDE.md's "active safety filter" roadmap item in its most direct form (correct
unsafe actions via optimisation before execution). It uses the same guidance signal as the
diffusion-steering hook but applies it to a fixed action chunk, so it does not depend on the
diffusion policy being task-competent.

TWO GUARDS AGAINST THE DEGENERATE "SAFE BY DOING NOTHING" RESULT (the Pusher2D failure mode,
RESUME 7.43 -- a frozen policy scores a perfect safety record):
  1. Hard trust region: each corrected position may move at most --max-delta metres from the
     commanded one, so the filter can nudge but cannot stall or rewrite the trajectory.
  2. Task progress is measured, not assumed: peak lift of the target block and final endpoint
     error against the clean expert trajectory are recorded for both conditions. A safety
     improvement that destroys task progress is reported as a failure, not a win.
"""
import argparse
import glob
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("SIM_HEADLESS", "1")

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
                      nominal_divergence)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WM_HIST = 3
SOURCE_DIR = str(EPISODES_DIR)
LIFT_SUCCESS_M = 0.03


def block_z(name):
    bid = mujoco.mj_name2id(SIM.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return float(SIM.data.xpos[bid][2]) if bid != -1 else 0.0


def build_noisy_plan(waypoints, noise_pos):
    """Same construction replay_noisy.py uses: fixed Gaussian noise on the commanded position,
    gripper left untouched (noising it just drops the block, per that script's own comment)."""
    plan = []
    for wp in waypoints:
        plan.append({
            "pos": np.array(wp["position"], dtype=np.float64) + np.random.normal(0, noise_pos, 3),
            "quat": np.array(wp["orientation"], dtype=np.float64),
            "grip": bool(wp["gripper"]),
            "t": wp["timestamp"],
            "clean_pos": np.array(wp["position"], dtype=np.float64),
        })
    return plan


def correct_chunk(world_model, cam_hist, prop_hist, act_hist, chunk_pos, chunk_grip,
                  n_opt_steps, lr, max_delta, mode="gradient", rng=None):
    """Gradient descent on a chunk of commanded positions to reduce predicted divergence,
    constrained to a hard trust region around the original commands.

    mode="random_matched" is the control that separates the two explanations for a negative
    result: it computes the gradient correction exactly as normal, then REPLACES it with a
    random direction of identical norm. If random-direction corrections are as harmful as
    gradient-directed ones, the finding is merely "perturbing commands by this magnitude
    hurts" (the corrections are ~3 mm against 2 mm of injected noise, so this is a real
    possibility). If gradient-directed corrections are meaningfully worse than random, the
    world model's divergence gradient is actively pointing the wrong way -- a much stronger
    claim about the metric itself."""
    visual = dino_preprocess(torch.from_numpy(np.stack(cam_hist)).unsqueeze(0), DEVICE)
    proprio = torch.from_numpy(np.stack(prop_hist)).float().unsqueeze(0).to(DEVICE)
    past = torch.from_numpy(np.stack(act_hist)).float().unsqueeze(0).to(DEVICE)

    orig = torch.tensor(chunk_pos, dtype=torch.float32, device=DEVICE)
    grip = torch.tensor(chunk_grip, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    delta = torch.zeros_like(orig, requires_grad=True)

    div0 = None
    for _ in range(n_opt_steps):
        acts = torch.cat([orig + delta, grip], dim=-1).unsqueeze(0)
        div = nominal_divergence(world_model, visual, proprio,
                                 torch.cat([past, acts], dim=1), DEVICE).mean()
        if div0 is None:
            div0 = float(div.detach())
        g = torch.autograd.grad(div, delta)[0]
        with torch.no_grad():
            delta -= lr * g
            delta.clamp_(-max_delta, max_delta)   # hard trust region: cannot stall the task
        delta.requires_grad_(True)

    with torch.no_grad():
        if mode == "random_matched":
            # same total magnitude, random direction -- isolates "wrong direction" from
            # "any perturbation of this size is harmful"
            r = torch.randn(delta.shape, device=delta.device,
                            generator=rng) if rng is not None else torch.randn_like(delta)
            delta = r / (r.norm() + 1e-12) * delta.norm()
            delta = delta.clamp(-max_delta, max_delta)
        acts = torch.cat([orig + delta, grip], dim=-1).unsqueeze(0)
        div1 = float(nominal_divergence(world_model, visual, proprio,
                                        torch.cat([past, acts], dim=1), DEVICE).mean())
        out = (orig + delta).cpu().numpy().astype(np.float64)
    return out, div0, div1


def run_trial(world_model, robot, cams, waypoints, seed, noise_pos, filtered,
              chunk_h, n_opt_steps, lr, max_delta, mode="gradient", video_out=None):
    np.random.seed(seed)
    plan = build_noisy_plan(waypoints, noise_pos)     # identical across conditions (same seed)
    np.random.seed(seed + 500_000)                    # separate stream so reset also matches
    robot.reset()
    time.sleep(0.5)

    start_z = block_z("block_middle")
    cam_hist, prop_hist, act_hist = (deque(maxlen=WM_HIST), deque(maxlen=WM_HIST),
                                     deque(maxlen=WM_HIST))
    toppled, topple_step, peak_tilt, peak_lift = False, None, 0.0, 0.0
    div_before, div_after, n_corr, total_shift = [], [], 0, 0.0
    video_frames = [] if video_out else None
    t0_real, t0_sim = time.time(), plan[0]["t"]

    def execute_step(idx, pos_cmd):
        """Execute one control step, capturing (frame, proprio) BEFORE commanding it so the
        history stays aligned: frame_k pairs with action_k, at consecutive 10 Hz steps.

        This alignment is the whole point -- an earlier version appended to cam_hist/prop_hist
        once per CHUNK while appending to act_hist once per STEP, so the world model received
        history frames 8 control steps apart (0.8 s, vs the consecutive steps it was trained
        on) paired with actions from a completely different point in time. That silently fed
        the divergence estimate out-of-distribution input, which is exactly what it is most
        sensitive to."""
        s = plan[idx]
        _, c2 = cams.get_frames()
        cam_hist.append(c2)
        prop_hist.append(robot.get_state())
        if video_frames is not None:
            video_frames.append(np.transpose(c2, (1, 2, 0)).copy())

        target_t = s["t"] - t0_sim
        elapsed = time.time() - t0_real
        if target_t > elapsed:
            time.sleep(target_t - elapsed)
        robot.update_desired_ee_pose(torch.Tensor(pos_cmd), torch.Tensor(s["quat"]))
        with SIM.lock:
            SIM.gripper_val = 0.0 if s["grip"] else 110
        act_hist.append(np.append(pos_cmd, 1.0 if s["grip"] else -1.0).astype(np.float32))
        return SIM.check_failure(TOPPLE_THRESHOLD_DEG)

    # prime the history with WM_HIST consecutive unfiltered steps so the first correction has
    # a properly-formed, in-distribution history to condition on
    i = 0
    while i < min(WM_HIST, len(plan)):
        failed, _, tilt = execute_step(i, plan[i]["pos"])
        peak_tilt = max(peak_tilt, tilt)
        peak_lift = max(peak_lift, block_z("block_middle") - start_z)
        if failed and not toppled:
            toppled, topple_step = True, i
        i += 1

    while i < len(plan):
        chunk = plan[i:i + chunk_h]
        pos = np.stack([s["pos"] for s in chunk])
        grip = np.array([1.0 if s["grip"] else -1.0 for s in chunk], dtype=np.float32)

        if filtered and len(cam_hist) == WM_HIST:
            pos_new, d0, d1 = correct_chunk(world_model, cam_hist, prop_hist, act_hist,
                                            pos, grip, n_opt_steps, lr, max_delta, mode=mode)
            total_shift += float(np.abs(pos_new - pos).mean())
            div_before.append(d0); div_after.append(d1); n_corr += 1
            pos = pos_new

        for j in range(len(chunk)):
            failed, _, tilt = execute_step(i + j, pos[j])
            peak_tilt = max(peak_tilt, tilt)
            peak_lift = max(peak_lift, block_z("block_middle") - start_z)
            if failed and not toppled:
                toppled, topple_step = True, i + j
        i += chunk_h

    if video_out and video_frames:
        import cv2 as _cv2, imageio.v2 as _imageio
        tag = ("FILTERED (world-model safety filter)" if filtered else "CONTROL (unmodified)")
        col = (200, 40, 40) if toppled else (30, 160, 30)
        w = _imageio.get_writer(video_out, fps=10)
        for k, fr in enumerate(video_frames):
            img = _cv2.resize(fr, (480, 360), interpolation=_cv2.INTER_NEAREST)
            img = _cv2.copyMakeBorder(img, 34, 26, 6, 6, _cv2.BORDER_CONSTANT, value=col)
            _cv2.putText(img, tag, (10, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                         (255, 255, 255), 1, _cv2.LINE_AA)
            status = "TOPPLED" if (toppled and topple_step is not None and k >= topple_step) else "ok"
            _cv2.putText(img, f"step {k:3d}   {status}", (10, img.shape[0] - 8),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv2.LINE_AA)
            w.append_data(img)
        w.close()

    final_pos = robot.get_state()[:3]
    endpoint_err = float(np.linalg.norm(final_pos - plan[-1]["clean_pos"]))
    return {
        "seed": seed, "filtered": bool(filtered), "mode": mode if filtered else "control",
        "toppled": bool(toppled), "topple_step": topple_step,
        "peak_tilt_deg": round(peak_tilt, 3),
        "peak_lift_m": round(float(peak_lift), 4),
        "picked": bool(peak_lift > LIFT_SUCCESS_M),
        "endpoint_err_m": round(endpoint_err, 4),
        "mean_abs_correction_m": round(total_shift / max(n_corr, 1), 5) if filtered else 0.0,
        "div_before": round(float(np.mean(div_before)), 5) if div_before else None,
        "div_after": round(float(np.mean(div_after)), 5) if div_after else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--noise-pos", type=float, default=0.002)
    ap.add_argument("--chunk-h", type=int, default=8)
    ap.add_argument("--opt-steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mode", choices=["gradient", "random_matched"], default="gradient",
                    help="random_matched = same correction magnitude, random direction "
                         "(control separating 'wrong direction' from 'any meddling hurts')")
    ap.add_argument("--max-delta", type=float, default=0.005,
                    help="trust region in metres; filter may not move a command further")
    ap.add_argument("--episodes", type=int, nargs="+", default=[1],
                    help="source expert trajectories, cycled across trials; a pair always "
                         "shares the same source episode so the comparison stays paired")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--out", default="results/steering_replay.jsonl")
    args = ap.parse_args()

    if not Path(args.out).is_absolute():
        args.out = str(Path(__file__).resolve().parent.parent / args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wp_by_ep = {}
    for ep in args.episodes:
        src = sorted(glob.glob(f"{SOURCE_DIR}/{ep}/*.json"))[0]
        wp_by_ep[ep] = json.load(open(src))["waypoints"]
    print(f"source episodes {args.episodes} "
          f"({[len(w) for w in wp_by_ep.values()]} waypoints), "
          f"noise_pos={args.noise_pos} m, trust region={args.max_delta} m", flush=True)

    world_model = load_frozen_world_model(DEVICE)
    robot = SimRobotInterface()
    cams = SimDualCamera()

    fout = open(args.out, "a")
    rows = []
    for k in range(args.n_trials):
        seed = args.seed0 + k
        ep = args.episodes[k % len(args.episodes)]
        waypoints = wp_by_ep[ep]
        for filtered in (False, True):
            r = run_trial(world_model, robot, cams, waypoints, seed, args.noise_pos,
                          filtered, args.chunk_h, args.opt_steps, args.lr, args.max_delta,
                          mode=args.mode)
            r["source_episode"] = ep
            rows.append(r)
            fout.write(json.dumps(r) + "\n"); fout.flush()
            tag = (args.mode[:8].upper() if filtered else "CONTROL ")
            extra = (f" corr={r['mean_abs_correction_m']*1000:.2f}mm "
                     f"div {r['div_before']}->{r['div_after']}") if filtered else ""
            print(f"[{k+1}/{args.n_trials} seed={seed}] {tag} toppled={r['toppled']!s:<5} "
                  f"tilt={r['peak_tilt_deg']:6.2f} lift={r['peak_lift_m']:.3f} "
                  f"endpt_err={r['endpoint_err_m']:.3f}{extra}", flush=True)
    fout.close()

    def agg(f):
        s = [r for r in rows if r["filtered"] == f]
        n = max(len(s), 1)
        return dict(topple_rate=sum(r["toppled"] for r in s) / n,
                    pick_rate=sum(r["picked"] for r in s) / n,
                    mean_peak_tilt=float(np.mean([r["peak_tilt_deg"] for r in s])) if s else float("nan"),
                    mean_endpoint_err=float(np.mean([r["endpoint_err_m"] for r in s])) if s else float("nan"),
                    n=len(s))

    c, f = agg(False), agg(True)
    paired = {}
    for r in rows:
        paired.setdefault(r["seed"], {})[r["filtered"]] = r
    both = [v for v in paired.values() if len(v) == 2]
    prevented = sum(1 for v in both if v[False]["toppled"] and not v[True]["toppled"])
    caused = sum(1 for v in both if not v[False]["toppled"] and v[True]["toppled"])

    print(f"\n{'='*74}\nRESULTS -- noisy expert replay in MuJoCo, {len(both)} paired trials\n{'='*74}")
    print(f"  CONTROL   topple {c['topple_rate']:.1%}  picked {c['pick_rate']:.1%}  "
          f"tilt {c['mean_peak_tilt']:.2f} deg  endpoint err {c['mean_endpoint_err']:.3f} m  (n={c['n']})")
    print(f"  FILTERED  topple {f['topple_rate']:.1%}  picked {f['pick_rate']:.1%}  "
          f"tilt {f['mean_peak_tilt']:.2f} deg  endpoint err {f['mean_endpoint_err']:.3f} m  (n={f['n']})")
    print(f"  paired: filter PREVENTED {prevented} topple(s), CAUSED {caused}")
    print("  NOTE: a topple-rate win is only real if pick rate and endpoint error hold up --")
    print("        a filter that stalls the arm would look perfectly safe.")
    json.dump({"control": c, "filtered": f, "paired_prevented": prevented,
               "paired_caused": caused, "n_paired": len(both), "args": vars(args)},
              open(str(Path(args.out).with_suffix(".summary.json")), "w"), indent=1)


if __name__ == "__main__":
    main()
