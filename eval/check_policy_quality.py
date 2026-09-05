"""
Is the trained Jenga policy actually any good, and was the logged `train_action_mse_error`
metric measuring it correctly?

Motivation: the training logs show `train_action_mse_error` flat at ~0.38-0.41 from epoch 0
to 60 while the denoising loss fell 0.26 -> 0.018. Suspected cause is a config bug rather than
a bad policy: `train_franka_dual_jenga.yaml` sets `num_inference_steps: 100` while the DDIM
scheduler has `num_train_timesteps: 50`. DDIM computes
`step_ratio = num_train_timesteps // num_inference_steps` = 50 // 100 = **0**, so every
sampling timestep collapses to 0 and the sampler is meaningless -- which is what the periodic
in-training sampling eval used.

This measures action MSE against ground truth across a sweep of num_inference_steps to
separate "policy is bad" from "the metric was computed with a broken sampler".
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_external_paths; add_external_paths(dino_wm=True, diffusion_policy=True)
from policy_loader import load_diffusion_policy, DEVICE  # noqa: E402
from diffusion_policy.dataset.jenga_dual_dataset import JengaDualDataset  # noqa: E402


def main():
    policy = load_diffusion_policy()
    ds = JengaDualDataset(
        dataset_path="/home/sanger/wksp/diffusion_policy/data/jenga_muj_imp.zarr",
        horizon=16, pad_before=1, pad_after=7, seed=42, val_ratio=0.05)
    print(f"dataset: {len(ds)} sequences")

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), 24, replace=False)
    batch = [ds[int(i)] for i in idxs]
    obs = {k: torch.stack([b["obs"][k] for b in batch]).to(DEVICE) for k in batch[0]["obs"]}
    gt = torch.stack([b["action"] for b in batch]).to(DEVICE)

    # policy consumes only the first n_obs_steps of the sequence
    To = policy.n_obs_steps
    obs_in = {k: v[:, :To] for k, v in obs.items()}

    print(f"\n{'steps':>8}{'action MSE':>14}{'per-dim RMSE (x,y,z,grip)':>34}")
    print(f"{'-'*56}")
    for n_steps in (100, 50, 32, 16, 8, 4):
        policy.num_inference_steps = n_steps
        with torch.no_grad():
            pred = policy.predict_action(obs_in)["action_pred"]
        mse = torch.nn.functional.mse_loss(pred, gt).item()
        rmse = torch.sqrt(((pred - gt) ** 2).mean(dim=(0, 1))).cpu().numpy()
        flag = "  <-- config default (BROKEN: > num_train_timesteps=50)" if n_steps == 100 else ""
        print(f"{n_steps:>8}{mse:>14.5f}   [{', '.join(f'{v:.4f}' for v in rmse)}]{flag}")

    # scale reference: how far does the expert actually move within one chunk? An MSE of X is
    # only meaningful against the magnitude of the motion being predicted.
    motion = (gt[:, 1:] - gt[:, :-1]).abs().mean(dim=(0, 1)).cpu().numpy()
    spread = gt.std(dim=(0, 1)).cpu().numpy()
    print(f"\n  reference -- mean |step-to-step delta| per dim: "
          f"[{', '.join(f'{v:.4f}' for v in motion)}]")
    print(f"  reference -- std of ground-truth actions per dim: "
          f"[{', '.join(f'{v:.4f}' for v in spread)}]")
    print("  (a policy predicting the per-dim mean would score MSE ~= mean of variances = "
          f"{float((gt.std(dim=(0,1))**2).mean()):.5f})")


if __name__ == "__main__":
    main()
