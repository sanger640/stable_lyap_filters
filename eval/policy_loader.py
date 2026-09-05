"""
First offline validation of the steering hook (section 8.2/8.3): on a handful of real
held-out observations, compare unsteered vs. steered action-chunk sampling from the trained
Jenga diffusion policy, using the SAME random seed for both so any difference is attributable
to guidance, not resampling noise. Checks:
  1. Does guidance measurably reduce the world model's own divergence score on its own output?
  2. How much does the steered action differ from the unsteered one (sanity: not degenerate)?
  3. Any NaN/exploding gradients?

This is a cheap, offline sanity check before attempting a live MuJoCo rollout comparison
(much more engineering, only worth it if this passes).
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_external_paths; add_external_paths(dino_wm=True, diffusion_policy=True)
from steering import (load_frozen_world_model, dino_preprocess, nominal_divergence,
                      steered_predict_action, ACTION_MEAN, ACTION_STD,
                      PROPRIO_MEAN, PROPRIO_STD)  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# absolute: this module is imported from panda_express/ too (eval_steering_sim.py), where a
# path relative to diffusion_policy/ would not resolve
CKPT = ("/home/sanger/wksp/diffusion_policy/data/outputs/2026.08.11/"
        "21.59.04_train_franka_dual_jenga_jenga_image_dual/checkpoints/"
        "epoch=0060-train_loss=0.018.ckpt")
from paths import EPISODES_DIR  # noqa: E402
N_OBS_STEPS = 2


def load_diffusion_policy():
    with hydra.initialize(config_path="diffusion_policy/config", version_base=None):
        cfg = hydra.compose(config_name="train_franka_dual_jenga",
                            overrides=["policy.down_dims=[128,256,512]"])
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    payload = torch.load(open(CKPT, "rb"), pickle_module=dill, map_location=DEVICE)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(DEVICE)
    policy.eval()
    policy.num_inference_steps = 16  # cut from config's 100 for iteration speed; revisit
    return policy


def load_real_obs(ep_dir, start_idx=20):
    """Pull N_OBS_STEPS consecutive real frames + proprio from a raw episode directory."""
    json_files = glob.glob(str(Path(ep_dir) / "*.json"))
    data = json.load(open(json_files[0]))
    wps = data["waypoints"][start_idx:start_idx + N_OBS_STEPS]
    rgb_dir = Path(ep_dir) / "rgb_frames"
    cam1_files = sorted(glob.glob(str(rgb_dir / "cam1_*.png")))
    cam2_files = sorted(glob.glob(str(rgb_dir / "cam2_*.png")))
    img_ts = np.array([float(Path(f).stem.split("_")[1]) / 1000.0 for f in cam1_files])

    cam1_imgs, cam2_imgs, pos4 = [], [], []
    for wp in wps:
        idx = int(np.argmin(np.abs(img_ts - wp["timestamp"])))
        im1 = cv2.cvtColor(cv2.imread(cam1_files[idx]), cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(cv2.imread(cam2_files[idx]), cv2.COLOR_BGR2RGB)
        cam1_imgs.append(cv2.resize(im1, (320, 240)))
        cam2_imgs.append(cv2.resize(im2, (320, 240)))
        pos4.append(list(wp["proc_pos"]) + [float(wp["proc_gripper"])])

    cam1 = np.stack(cam1_imgs).transpose(0, 3, 1, 2)  # (T,3,H,W)
    cam2 = np.stack(cam2_imgs).transpose(0, 3, 1, 2)
    pos4 = np.array(pos4, dtype=np.float32)
    return cam1, cam2, pos4


def main():
    print("loading frozen world model ...", flush=True)
    world_model = load_frozen_world_model(DEVICE)
    print("loading diffusion policy ...", flush=True)
    policy = load_diffusion_policy()

    episode_dirs = sorted(glob.glob(f"{EPISODES_DIR}/*"), key=lambda p: int(Path(p).name)
                          if Path(p).name.isdigit() else 0)
    test_eps = [d for d in episode_dirs if Path(d).name.isdigit()][:5]

    for ep_dir in test_eps:
        print(f"\n=== episode {Path(ep_dir).name} ===", flush=True)
        try:
            cam1, cam2, pos4 = load_real_obs(ep_dir, start_idx=20)
        except Exception as e:
            print(f"  skip ({e})")
            continue

        cam1_t = torch.from_numpy(cam1).float().unsqueeze(0).to(DEVICE) / 255.0
        cam2_t = torch.from_numpy(cam2).float().unsqueeze(0).to(DEVICE) / 255.0
        pos_t = torch.from_numpy(pos4).float().unsqueeze(0).to(DEVICE)
        obs_dict = {"camera_1": cam1_t, "camera_2": cam2_t, "agent_pos": pos_t}

        # dino_wm side: cam2 only (single-view checkpoint), its own 224x224 preprocessing
        visual_dino = dino_preprocess(torch.from_numpy(cam2).unsqueeze(0), DEVICE)
        proprio_dino = pos_t  # same [x,y,z,gripper] convention on both sides

        gen_seed = 0
        for guidance_scale in (0.0, 0.5, 2.0, 5.0):
            gen = torch.Generator(device=DEVICE).manual_seed(gen_seed)
            with torch.enable_grad():
                result = steered_predict_action(policy, obs_dict, world_model,
                                                 visual_dino, proprio_dino,
                                                 guidance_scale=guidance_scale,
                                                 guidance_frac=0.3, generator=gen)
            action = result["action"][0].detach().cpu().numpy()
            divs = result["guidance_divergences"]
            with torch.no_grad():
                final_div = nominal_divergence(world_model, visual_dino, proprio_dino,
                                               result["action_pred"][:, :, :4], DEVICE).item()
            print(f"  guidance_scale={guidance_scale:4.1f}  "
                  f"final_action[0]={action[0]}  "
                  f"guided_div_trace={[f'{d:.4f}' for d in divs][:3]}...{[f'{d:.4f}' for d in divs][-3:]}  "
                  f"final_div={final_div:.5f}", flush=True)


if __name__ == "__main__":
    main()
