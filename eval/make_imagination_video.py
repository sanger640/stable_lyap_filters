"""
Side-by-side videos of UNSTEERED vs STEERED action chunks (section 8.4), so the effect of
guidance can be judged visually rather than only through the scalar divergence number.

What each frame shows, left to right:
  1. REAL OBSERVATION   -- the actual camera view the policy conditioned on (static)
  2. UNSTEERED PREDICTION -- the world model's decoded imagination of the future under the
                             policy's own action chunk (guidance_scale=0), animating over the
                             prediction horizon
  3. STEERED PREDICTION   -- same, under the guided action chunk (guidance_scale>0), SAME
                             random seed so the only difference is guidance
  4. |DIFFERENCE|         -- per-pixel absolute difference between 2 and 3, amplified, so
                             small corrections are actually visible
  5. ACTION TRAJECTORY    -- the two action chunks plotted against each other

IMPORTANT correctness fix vs. the first steering tests (sections 8.2-8.3): those fed the
world model only 2 history frames because that's what the diffusion policy uses
(`n_obs_steps=2`), but dino_wm's predictor was TRAINED with `num_hist=3`. It doesn't crash --
ViTPredictor does `x + self.pos_embedding[:, :n]`, silently slicing the position embedding to
whatever length it gets -- so the mismatch was invisible. This script gives the world model
its proper 3 frames (and the 2 real past actions that precede the chunk, taken from the
recorded episode, matching how test_monitor.py aligns actions with history frames), while the
diffusion policy still gets its own 2. Re-measured divergences therefore differ slightly from
sections 8.2/8.3's numbers.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_external_paths; add_external_paths(dino_wm=True, diffusion_policy=True)
from steering import (load_frozen_world_model, dino_preprocess, nominal_divergence,  # noqa: E402
                      steered_conditional_sample, ACTION_MEAN, ACTION_STD,
                      PROPRIO_MEAN, PROPRIO_STD)
from policy_loader import load_diffusion_policy, DEVICE  # noqa: E402

WM_HIST = 3       # dino_wm's trained num_hist
DP_HIST = 2       # diffusion policy's n_obs_steps
PANEL = 224
from paths import EPISODES_DIR  # noqa: E402


def load_obs_window(ep_dir, start_idx):
    """Return WM_HIST consecutive frames (cam1+cam2), their proprio, and the actions taken
    at each of those frames (needed so the world model's rollout gets past actions aligned
    with its history frames, exactly like test_monitor.py does)."""
    data = json.load(open(glob.glob(str(Path(ep_dir) / "*.json"))[0]))
    wps = data["waypoints"]
    if start_idx + WM_HIST + 2 >= len(wps):
        raise ValueError("window past end of episode")
    win = wps[start_idx:start_idx + WM_HIST]

    rgb_dir = Path(ep_dir) / "rgb_frames"
    cam1_files = sorted(glob.glob(str(rgb_dir / "cam1_*.png")))
    cam2_files = sorted(glob.glob(str(rgb_dir / "cam2_*.png")))
    img_ts = np.array([float(Path(f).stem.split("_")[1]) / 1000.0 for f in cam1_files])

    cam1, cam2, proprio, past_actions = [], [], [], []
    for wp in win:
        i = int(np.argmin(np.abs(img_ts - wp["timestamp"])))
        cam1.append(cv2.resize(cv2.cvtColor(cv2.imread(cam1_files[i]), cv2.COLOR_BGR2RGB), (320, 240)))
        cam2.append(cv2.resize(cv2.cvtColor(cv2.imread(cam2_files[i]), cv2.COLOR_BGR2RGB), (320, 240)))
        proprio.append(list(wp["proc_pos"]) + [1.0 if wp["proc_gripper"] else -1.0])
        # "action at this frame" = the commanded pose recorded at this waypoint
        past_actions.append(list(wp["position"]) + [1.0 if wp["gripper"] else -1.0])

    return (np.stack(cam1).transpose(0, 3, 1, 2), np.stack(cam2).transpose(0, 3, 1, 2),
            np.array(proprio, np.float32), np.array(past_actions, np.float32))


def rollout_and_decode(world_model, visual_dino, proprio_dino, past_actions, chunk, device):
    """Roll the world model forward under [past_actions ++ chunk] and decode every predicted
    latent back to pixels. Returns (decoded_uint8 (T,H,W,3), divergence float)."""
    am, asd = ACTION_MEAN.to(device), ACTION_STD.to(device)
    pm, psd = PROPRIO_MEAN.to(device), PROPRIO_STD.to(device)
    acts_real = torch.cat([past_actions, chunk], dim=1)
    acts_n = (acts_real - am) / asd
    proprio_n = (proprio_dino - pm) / psd
    obs0 = {"visual": visual_dino, "proprio": proprio_n}
    with torch.no_grad():
        z_obses, _ = world_model.rollout(obs0, acts_n)
        decoded, _ = world_model.decode_obs(z_obses)
        vis = decoded["visual"][0].cpu().numpy()          # (T,C,H,W) in [-1,1]
        vis = np.clip((vis + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        z_visual = z_obses["visual"]
        cos = torch.nn.functional.cosine_similarity
        from steering import build_patch_keep_mask
        d_end = 1 - cos(z_visual[:, WM_HIST], z_visual[:, -1], dim=-1)
        keep = build_patch_keep_mask(d_end.shape[-1], device)
        div = float(d_end[:, keep].mean())
    return vis, div


def label(img, text, color=(255, 255, 255), bg=(0, 0, 0), y=18):
    out = img.copy()
    cv2.rectangle(out, (0, y - 15), (out.shape[1], y + 6), bg, -1)
    cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def action_plot(unst, steer, t_now, size=PANEL):
    """Render the two action chunks as an image panel, with a marker at the current step."""
    # taller than the final panel then downscaled -- 4 stacked subplots plus a legend do not
    # fit legibly in a square figure (the first subplot's ylabel/legend got clipped)
    fig, axes = plt.subplots(4, 1, figsize=(size / 100, size / 62), dpi=100, sharex=True)
    names = ["x", "y", "z", "grip"]
    for k, ax in enumerate(axes):
        ax.plot(unst[:, k], color="tab:blue", lw=1.2, label="unsteered" if k == 0 else None)
        ax.plot(steer[:, k], color="tab:red", lw=1.2, ls="--", label="steered" if k == 0 else None)
        ax.axvline(t_now, color="0.5", lw=0.8)
        ax.set_ylabel(names[k], fontsize=6)
        ax.tick_params(labelsize=5)
    axes[0].legend(fontsize=5, loc="upper right")
    axes[-1].set_xlabel("horizon step", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.resize(buf, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--episodes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--start-idx", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/steering_videos")
    args = ap.parse_args()

    world_model = load_frozen_world_model(DEVICE)
    policy = load_diffusion_policy()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = []

    for ep in args.episodes:
        ep_dir = f"{EPISODES_DIR}/{ep}"
        try:
            cam1, cam2, proprio, past_actions = load_obs_window(ep_dir, args.start_idx)
        except Exception as e:
            print(f"episode {ep}: skip ({e})")
            continue

        # world model gets its full trained history (3 frames); policy gets its own last 2
        visual_dino = dino_preprocess(torch.from_numpy(cam2).unsqueeze(0), DEVICE)
        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(DEVICE)
        past_act_t = torch.from_numpy(past_actions).unsqueeze(0).to(DEVICE)

        obs_dict = {
            "camera_1": torch.from_numpy(cam1[-DP_HIST:]).float().unsqueeze(0).to(DEVICE) / 255.0,
            "camera_2": torch.from_numpy(cam2[-DP_HIST:]).float().unsqueeze(0).to(DEVICE) / 255.0,
            "agent_pos": proprio_t[:, -DP_HIST:],
        }

        nobs = policy.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        with torch.no_grad():
            global_cond = policy.obs_encoder(this_nobs).reshape(B, -1)
        cond_data = torch.zeros((B, policy.horizon, policy.action_dim), device=DEVICE)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        chunks = {}
        for name, scale in (("unsteered", 0.0), ("steered", args.guidance_scale)):
            gen = torch.Generator(device=DEVICE).manual_seed(0)
            with torch.enable_grad():
                nsample, _ = steered_conditional_sample(
                    policy, cond_data, cond_mask, world_model, visual_dino,
                    proprio_t, guidance_scale=scale, global_cond=global_cond, generator=gen)
            chunks[name] = policy.normalizer["action"].unnormalize(nsample[..., :policy.action_dim])

        vids, divs = {}, {}
        for name, chunk in chunks.items():
            vids[name], divs[name] = rollout_and_decode(
                world_model, visual_dino, proprio_t, past_act_t, chunk, DEVICE)

        pct = 100.0 * (divs["unsteered"] - divs["steered"]) / max(divs["unsteered"], 1e-9)
        print(f"episode {ep}: unsteered div={divs['unsteered']:.5f}  "
              f"steered div={divs['steered']:.5f}  ({pct:+.1f}%)", flush=True)
        summary.append({"episode": ep, "unsteered": divs["unsteered"],
                        "steered": divs["steered"], "pct_change": pct})

        obs_panel = cv2.resize(cam2[-1].transpose(1, 2, 0), (PANEL, PANEL))
        unst_np = chunks["unsteered"][0].detach().cpu().numpy()
        steer_np = chunks["steered"][0].detach().cpu().numpy()

        T = min(len(vids["unsteered"]), len(vids["steered"]))
        out_path = f"{args.out_dir}/ep{ep}_scale{args.guidance_scale:g}.mp4"
        writer = imageio.get_writer(out_path, fps=3)
        for t in range(T):
            a = cv2.resize(vids["unsteered"][t], (PANEL, PANEL))
            b = cv2.resize(vids["steered"][t], (PANEL, PANEL))
            d = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 6, 0, 255).astype(np.uint8)
            phase = "history (observed)" if t < WM_HIST else f"predicted +{t - WM_HIST + 1}"
            frames = [
                label(obs_panel, "REAL OBSERVATION (cam2)"),
                label(a, f"UNSTEERED  div={divs['unsteered']:.4f}", bg=(120, 60, 0)),
                label(b, f"STEERED s={args.guidance_scale:g}  div={divs['steered']:.4f}",
                      bg=(0, 60, 120)),
                label(d, "|DIFFERENCE| x6"),
                label(action_plot(unst_np, steer_np, t), "ACTION CHUNK"),
            ]
            row = np.concatenate(frames, axis=1)
            row = label(row, f"episode {ep}   frame {t}/{T - 1}   {phase}   "
                             f"divergence change {pct:+.1f}%", y=row.shape[0] - 6,
                        bg=(40, 40, 40))
            writer.append_data(row)
        writer.close()
        print(f"  -> {out_path}", flush=True)

    if summary:
        print("\n=== SUMMARY (world model given its correct 3-frame history) ===")
        for s in summary:
            print(f"  ep {s['episode']:>3}: {s['unsteered']:.5f} -> {s['steered']:.5f}  "
                  f"({s['pct_change']:+.1f}%)")
        improved = sum(1 for s in summary if s["pct_change"] > 0)
        print(f"  improved: {improved}/{len(summary)}")
        json.dump(summary, open(f"{args.out_dir}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
