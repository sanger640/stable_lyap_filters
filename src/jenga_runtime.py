"""Self-contained runtime primitives for the bundled Jenga world model and replay LMDB.

This module is deliberately independent of the old calibrated monitor in ``src/monitor.py``.
It establishes the data/model contract needed by PLAN_JENGA J1-J5 while keeping the vendored
dino_wm implementation unchanged.
"""
from dataclasses import dataclass
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LMDB = ROOT / "data" / "jenga" / "jenga_single_100.lmdb"
DEFAULT_CHECKPOINT = ROOT / "data" / "jenga" / "world_model.pt"
NUM_HIST = 3
ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114])
ACTION_STD = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065])
PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302])
PROPRIO_STD = torch.tensor([0.03217997, 0.01056713, 0.0327194, 0.4139551])


@dataclass(frozen=True)
class JengaEpisode:
    episode_id: str
    frame_keys: tuple
    actions: np.ndarray
    proprio: np.ndarray

    @property
    def length(self):
        return len(self.frame_keys)


class JengaReplay:
    """Read-only adapter around the exact LMDB schema in the transfer bundle."""

    def __init__(self, path=DEFAULT_LMDB):
        import lmdb
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False,
                             max_readers=64)
        with self.env.begin() as txn:
            raw = txn.get(b"__metadata__")
            if raw is None:
                raise ValueError(f"{self.path} has no __metadata__ record")
            self.metadata = pickle.loads(raw)
        self.episode_ids = tuple(sorted(self.metadata["episodes"], key=int))

    def close(self):
        self.env.close()

    def episode(self, episode_id):
        ep = str(episode_id)
        meta = self.metadata["episodes"][ep]
        with self.env.begin() as txn:
            actions = pickle.loads(_required(txn, f"{ep}_actions"))
            proprio = pickle.loads(_required(txn, f"{ep}_proprio"))
        keys = tuple(meta["keys"]["cam2"])
        n = int(meta["seq_len"])
        if len(keys) != n or actions.shape != (n, 4) or proprio.shape != (n, 4):
            raise ValueError(
                f"episode {ep} alignment error: seq_len={n}, frames={len(keys)}, "
                f"actions={actions.shape}, proprio={proprio.shape}")
        return JengaEpisode(ep, keys, np.asarray(actions, np.float32),
                            np.asarray(proprio, np.float32))

    def frames(self, episode, indices):
        ep = episode if isinstance(episode, JengaEpisode) else self.episode(episode)
        out = []
        with self.env.begin() as txn:
            for i in indices:
                raw = _required(txn, ep.frame_keys[int(i)])
                out.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
        return np.stack(out)


def _required(txn, key):
    key = key.encode() if isinstance(key, str) else key
    value = txn.get(key)
    if value is None:
        raise KeyError(key.decode(errors="replace"))
    return value


def preprocess_frames(frames, device="cpu", size=224):
    """HWC/CHW uint8 frames -> dino_wm's trained [-1,1] BCHW sequence."""
    from torchvision.transforms import functional as TF

    x = torch.as_tensor(np.asarray(frames), device=device)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5:
        raise ValueError(f"expected (T,H,W,3) or (B,T,H,W,3), got {tuple(x.shape)}")
    if x.shape[-1] == 3:
        x = x.permute(0, 1, 4, 2, 3)
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    b, t = x.shape[:2]
    x = TF.resize(x.reshape(b * t, 3, *x.shape[-2:]), [size, size], antialias=True)
    return (x.reshape(b, t, 3, size, size) - 0.5) / 0.5


def load_world_model(checkpoint=DEFAULT_CHECKPOINT, device="cuda", encoder=None):
    """Reconstruct VWorldModel around the four inference modules in the stripped checkpoint."""
    model_dir = str(ROOT / "src" / "models")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from dinowm import install_import_shim
    from dinowm.dino import DinoV2Encoder
    from dinowm.visual_world_model import VWorldModel

    install_import_shim()
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    required = {"predictor", "action_encoder", "proprio_encoder", "decoder", "epoch"}
    missing = required - set(ckpt)
    if missing:
        raise ValueError(f"checkpoint is missing {sorted(missing)}")
    if encoder is None:
        encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    model = VWorldModel(
        image_size=224, num_hist=NUM_HIST, num_pred=1, encoder=encoder,
        proprio_encoder=ckpt["proprio_encoder"], action_encoder=ckpt["action_encoder"],
        decoder=ckpt["decoder"], predictor=ckpt["predictor"], proprio_dim=10,
        action_dim=10, concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=False, train_decoder=False,
    )
    # The vendored VWorldModel.eval() predates nn.Module's fluent convention and returns None.
    # Keep the calls separate rather than chaining `.to(...).eval()`.
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.checkpoint_epoch = int(ckpt["epoch"])
    return model


def normalise_actions(actions, device):
    x = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
    return (x - ACTION_MEAN.to(device)) / ACTION_STD.to(device)


def normalise_proprio(proprio, device):
    x = torch.as_tensor(np.asarray(proprio), dtype=torch.float32, device=device)
    return (x - PROPRIO_MEAN.to(device)) / PROPRIO_STD.to(device)


def coherent_action_probes(actions, n=50, eps=0.10, seed=0):
    """Probe displacement of an absolute EE path; never perturb the binary gripper."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"expected actions shaped (T,4), got {a.shape}")
    direction = a - a[:1]
    direction[:, 3] = 0.0
    z = np.random.default_rng(seed).standard_normal((n, 1, 1)).astype(np.float32)
    return a[None] + eps * z * direction[None]
