"""
The safety monitor, end to end.

Given a short observation history and a proposed action chunk, score how unsafe the chunk is.
Higher score = less safe. Threshold on percentiles of the SAFE score distribution only, which
is what keeps the method zero-shot: no failure labels are ever used to fit or calibrate.

    from monitor import Monitor
    mon = Monitor.load()                       # frozen DINO world model
    mon.calibrate(safe_scores)                 # percentiles of safe chunks only
    score = mon.score(frames, proprio, actions)
    if score > mon.threshold: halt()

Default configuration is the one that won on held-out validation:
`d_end`, p90 over patches, row mask + PC1 background mask at 75% keep (AUC 0.894).
See README for the full comparison and metrics.py for why each piece is there.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DINO_WM_DIR, DINO_WM_CKPT, add_external_paths  # noqa: E402
import metrics  # noqa: E402

add_external_paths(dino_wm=True)
from server_single_max import load_model as _load_world_model  # noqa: E402

IMG_SIZE = 224
NUM_HIST = 3          # the predictor was TRAINED with 3; feeding fewer silently degrades it
# dataset normalisation constants (fallback when dataset_stats.pt is absent)
ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114])
ACTION_STD = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065])
PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302])
PROPRIO_STD = torch.tensor([0.03217997, 0.01056713, 0.0327194, 0.4139551])


def preprocess_frames(frames, device):
    """(T,3,H,W) uint8 or float -> (1,T,3,224,224) normalised the way the encoder expects."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    x = torch.as_tensor(np.asarray(frames)).float().to(device)
    if x.max() > 1.5:
        x = x / 255.0
    t = x.shape[0]
    return tf(x).reshape(1, t, 3, IMG_SIZE, IMG_SIZE)


class Monitor:
    def __init__(self, model, device="cuda", metric="d_end", reduce="p90",
                 mask="pc1", low_norm_k=30, pc1_keep_pct=75):
        self.model, self.device = model, device
        self.metric, self.reduce, self.mask_kind = metric, reduce, mask
        self.low_norm_k, self.pc1_keep_pct = low_norm_k, pc1_keep_pct
        self.row_keep = metrics.build_patch_keep_mask(device=device)
        self.pc1_mu = self.pc1_dir = self.pc1_sign = None
        self.threshold = None

    @classmethod
    def load(cls, device="cuda", **kw):
        import hydra
        with hydra.initialize_config_dir(config_dir=str(DINO_WM_DIR / "conf"), version_base=None):
            cfg = hydra.compose(config_name="train")
        model = _load_world_model(DINO_WM_CKPT, cfg, device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device, **kw)

    # ---- world model rollout -------------------------------------------------------
    def _rollout(self, frames, proprio, actions, n_perturb=0, noise_std=0.05, seed=0):
        """actions must cover the history steps AND the future: (NUM_HIST + horizon, 4), in
        real units. This alignment (action_k belongs to frame_k) is easy to get wrong and
        fails silently -- the predictor slices its position embedding to whatever it gets."""
        dev = self.device
        vis = preprocess_frames(frames, dev)
        assert vis.shape[1] == NUM_HIST, f"need {NUM_HIST} history frames, got {vis.shape[1]}"
        prop = torch.as_tensor(np.asarray(proprio)).float().to(dev).unsqueeze(0)
        act = torch.as_tensor(np.asarray(actions)).float().to(dev).unsqueeze(0)
        assert act.shape[1] > NUM_HIST, "actions must include history + future steps"

        if n_perturb:
            g = torch.Generator(device=dev).manual_seed(seed)
            vis = vis.repeat(n_perturb + 1, 1, 1, 1, 1)
            prop = prop.repeat(n_perturb + 1, 1, 1)
            act = act.repeat(n_perturb + 1, 1, 1).clone()
            act[1:, :, :3] += torch.randn(n_perturb, act.shape[1], 3,
                                          device=dev, generator=g) * noise_std

        prop_n = (prop - PROPRIO_MEAN.to(dev)) / PROPRIO_STD.to(dev)
        act_n = (act - ACTION_MEAN.to(dev)) / ACTION_STD.to(dev)
        with torch.no_grad():
            z, _ = self.model.rollout({"visual": vis, "proprio": prop_n}, act_n)
        return z["visual"]        # (B, NUM_HIST+H+1, P, F)

    # ---- masking -------------------------------------------------------------------
    def fit_pc1(self, safe_z_obs, motion=None):
        """Fit the PC1 direction on observed latents from SAFE chunks.

        `motion` (per-patch ground-truth pixel motion, same shape) resolves the sign. Pass it
        whenever you can: the sign is NOT determinable from the latents alone, and the
        norm-based heuristic that seems natural is empirically backwards (see metrics.pc1_mask).
        Without it we fall back to the validated convention, which may not hold for a new fit."""
        z = torch.as_tensor(np.asarray(safe_z_obs))
        self.pc1_mu, self.pc1_dir = metrics.fit_pc1_basis(z, self.row_keep.cpu())
        if motion is not None:
            proj = ((z[:, self.row_keep.cpu()].reshape(-1, z.shape[-1]).cpu().numpy()
                     - self.pc1_mu) @ self.pc1_dir)
            m = np.asarray(motion)[:, self.row_keep.cpu().numpy()].reshape(-1)
            c = float(np.corrcoef(proj, m)[0, 1])
            # keep LOW-motion (background) patches: they carry the cleaner signal
            self.pc1_sign = -1 if c > 0 else +1
        else:
            self.pc1_sign = +1
        return self

    def _mask(self, z_obs):
        if self.mask_kind == "row":
            return self.row_keep
        if self.mask_kind == "low_norm":
            return metrics.low_norm_mask(z_obs, self.row_keep, self.low_norm_k)
        if self.mask_kind == "pc1":
            if self.pc1_mu is None:
                raise RuntimeError("call fit_pc1() before using mask='pc1'")
            return metrics.pc1_mask(z_obs, self.row_keep, self.pc1_mu, self.pc1_dir,
                                    sign=self.pc1_sign, keep_pct=self.pc1_keep_pct)
        raise ValueError(self.mask_kind)

    # ---- scoring -------------------------------------------------------------------
    def score(self, frames, proprio, actions, n_perturb=None, noise_std=0.05, seed=0):
        """Scalar unsafety score for one action chunk. Higher = less safe."""
        need_pert = self.metric == "ftle_variance" if n_perturb is None else n_perturb
        n_pert = (50 if self.metric == "ftle_variance" else 0) if n_perturb is None else n_perturb
        z = self._rollout(frames, proprio, actions, n_pert, noise_std, seed)
        z_obs = z[0, NUM_HIST - 1]
        mask = self._mask(z_obs)
        if self.metric == "d_end":
            return metrics.d_end(z[0], NUM_HIST, mask, self.reduce)
        if self.metric == "ftle_variance":
            return metrics.ftle_variance(z[1:, -1], mask, self.reduce)
        raise ValueError(self.metric)

    def latents(self, frames, proprio, actions):
        """Observed latent for a chunk -- what you collect to fit PC1."""
        return self._rollout(frames, proprio, actions)[0, NUM_HIST - 1].cpu().numpy()

    # ---- calibration ---------------------------------------------------------------
    def calibrate(self, safe_scores, percentile=95):
        """Threshold from SAFE chunks only -- never uses failure labels.

        p95 was the best operating point found (recall .52, precision .13, 0.88 false alarms
        per episode). Do NOT report accuracy: at a 1.4% base rate, always predicting 'safe'
        scores 98.6% and beats every real configuration."""
        s = np.asarray(safe_scores, float)
        self.threshold = float(np.percentile(s[np.isfinite(s)], percentile))
        return self.threshold

    def is_unsafe(self, *a, **kw):
        if self.threshold is None:
            raise RuntimeError("call calibrate() first")
        return self.score(*a, **kw) > self.threshold
