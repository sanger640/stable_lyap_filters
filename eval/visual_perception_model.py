"""Step 1 perception: frozen DINOv2 (close fixed camera) + keypoint head -> block poses, and its error model.

Same head and training recipe as visual_encoder_precision_addendum_2 (the 2.54 mm result), but the
trained head is SAVED, and its held-out residuals define the perception error model the monitor
assumes: a full 9x9 covariance of block-position errors (keeps correlated errors such as a common
offset) plus a per-block 3x3 covariance of rotation-vector errors. Validation residuals only; no
task label. Also provides `perceive(frame)` for rendered frames of new scenes.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

import jenga_spatial_readout as sr  # noqa: E402

ENCODED = ROOT / "results/program_b/vjepa/enc_dinov2_392_close"
MODEL = ROOT / "results/program_b/perception/dinov2_close_keypoint.pt"
ERROR_MODEL = ROOT / "results/jenga/visual_perception_error_model.json"


def rotation_matrices(six):
    """(N, 3 blocks x 6) row-major first-two-columns -> (N, 3, 3, 3) orthonormal rotations."""
    six = np.asarray(six, float).reshape(len(six), 3, 3, 2)
    c0 = six[..., 0] / np.linalg.norm(six[..., 0], axis=-1, keepdims=True)
    c1 = six[..., 1] - (c0 * six[..., 1]).sum(-1, keepdims=True) * c0
    c1 /= np.linalg.norm(c1, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=-1)


def rotation_errors(pred_six, true_six):
    from scipy.spatial.transform import Rotation
    rp, rt = rotation_matrices(pred_six), rotation_matrices(true_six)
    rel = np.einsum("nbij,nbkj->nbik", rp, rt)                    # R_pred R_true^T
    return Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(len(rel), 3, 3)


def train_and_save(device="cuda", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    grids, states, ids = sr.load(ENCODED)
    unique = sorted(set(ids)); val = set(unique[:max(1, int(len(unique) * 0.12))])
    held = np.array([i in val for i in ids])
    y = torch.from_numpy(states[:, :sr.POSE])
    mean, std = y[~held].mean(0), y[~held].std(0).clamp_min(1e-6)
    model = sr.KeypointHead(grids.shape[-1]).to(device)
    x_train, y_train = torch.from_numpy(grids[~held]), (y[~held] - mean) / std
    steps = sr.EPOCHS * int(np.ceil(len(x_train) / sr.BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(sr.EPOCHS):
        order = torch.randperm(len(x_train))
        for first in range(0, len(order), sr.BATCH):
            idx = order[first:first + sr.BATCH]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x_train[idx].to(device).float())
            loss = ((pred.float() - y_train[idx].to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        x_held = torch.from_numpy(grids[held])
        pred = torch.cat([model(x_held[i:i + 256].to(device).float()).float().cpu()
                          for i in range(0, len(x_held), 256)]) * std + mean
    truth = states[held][:, :sr.POSE]
    pos_err = pred[:, :9].numpy() - truth[:, :9]
    rot_err = rotation_errors(pred[:, 9:27].numpy(), truth[:, 9:27])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "channels": grids.shape[-1]}, MODEL)
    error = {
        "source": "held-out residuals of the saved keypoint head (validation episodes, V1 TRAIN layouts)",
        "held_out_states": int(held.sum()),
        "position_bias_m": pos_err.mean(0).tolist(),
        "position_covariance_m2": np.cov(pos_err, rowvar=False).tolist(),
        "position_rmse_mm_per_block": (1000 * np.sqrt((pos_err.reshape(-1, 3, 3) ** 2).sum(-1).mean(0))).tolist(),
        "rotation_covariance_rad2_per_block": [np.cov(rot_err[:, b], rowvar=False).tolist() for b in range(3)],
        "rotation_rms_deg_per_block": np.degrees(np.sqrt((rot_err ** 2).sum(-1).mean(0))).tolist(),
        "geometry": sr.vp.geometry_errors(torch.from_numpy(np.concatenate(
            [pred.numpy(), states[held][:, sr.POSE:]], 1)), torch.from_numpy(states[held])),
    }
    ERROR_MODEL.write_text(json.dumps(error, indent=2) + "\n")
    return error


class Perceiver:
    """Rendered close-camera frame(s) -> estimated (3, 7) block poses (position, wxyz quaternion)."""

    def __init__(self, device="cuda"):
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        saved = torch.load(MODEL, map_location=device, weights_only=False)
        self.head = sr.KeypointHead(saved["channels"]).to(device).eval()
        self.head.load_state_dict(saved["state_dict"])
        self.mean, self.std = saved["mean"].to(device), saved["std"].to(device)
        self.encoder = load_world_model(DEFAULT_CHECKPOINT, device).encoder
        self.device = device

    def __call__(self, frames):
        import sys as _s
        _s.path.insert(0, str(ROOT / "eval"))
        from jenga_vjepa_encode import dino_tokens
        from scipy.spatial.transform import Rotation
        tokens = dino_tokens(self.encoder, np.asarray(frames), self.device)       # (N, 784, 384)
        grid = tokens.reshape(len(tokens), 28, 28, -1)
        with torch.no_grad():
            pose = (self.head(grid) * self.std + self.mean).cpu().numpy()
        rot = rotation_matrices(pose[:, 9:27])
        quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(len(pose), 3, 4)
        return np.concatenate([pose[:, :9].reshape(-1, 3, 3), quat[..., [3, 0, 1, 2]]], -1)


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in train_and_save().items()
                      if k in ("position_rmse_mm_per_block", "rotation_rms_deg_per_block", "geometry")}, indent=1))
