"""Phase 2 amendment 7: encoder smoothness diagnostic (DINOv2 vs iBOT) on the H2 renders from the V-JEPA test.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_7).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

FRAMES = ROOT / "results/program_b/vjepa_ac/frames"
RESULT = ROOT / "results/jenga/phase2_encoder_diag_result.json"
IBOT = Path.home() / ".cache/torch/hub/checkpoints/ibot_vits16_teacher.pth"
MONITOR_SEED = 20261230


class IBot:
    """iBOT ViT-S/16 teacher: (N, 384, 384, 3) uint8 -> (N, 784, 384) patch tokens at 448 px."""

    def __init__(self):
        import timm
        import torch
        sd = torch.load(IBOT, map_location="cpu")["state_dict"]
        sd = {k.replace("backbone.", ""): v for k, v in sd.items() if not k.startswith("head")}
        self.m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
        self.m.load_state_dict(sd, strict=True)
        self.m = self.m.cuda().eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, frames, batch=32):
        import torch
        from torchvision.transforms import functional as TF
        out = []
        for i in range(0, len(frames), batch):
            x = torch.as_tensor(frames[i:i + batch], device="cuda").permute(0, 3, 1, 2).float() / 255
            x = (TF.resize(x, [448, 448], antialias=True) - self.mean) / self.std
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self.m.forward_features(x)[:, 1:].float())
        return torch.cat(out)


def dino_tokens_fn():
    import step3_data as sd
    lat = sd.Latent()
    from jenga_vjepa_encode import dino_tokens

    def tok(frames):
        return dino_tokens(lat.encoder, np.asarray(frames), "cuda")
    return lat, tok


def fit_channel_pca(tokens, k=8):
    """tokens (M, C) -> (mean, basis (C,k), z_mean, z_std) as eval/step3_data channel PCA."""
    import torch
    mean = tokens.mean(0)
    _, _, v = torch.linalg.svd(tokens - mean, full_matrices=False)
    basis = v[:k].T
    z = (tokens - mean) @ basis
    return mean, basis, z.mean(0), z.std(0)


def main():
    import torch
    from scipy.stats import spearmanr
    from phase2_amend4 import alarm, score
    files = sorted(FRAMES.glob("*.npz"))
    lat, dtok = dino_tokens_fn()
    ibot = IBot()
    # PCA fits on the H2 true-start frames (diagnostic only)
    true_starts = np.stack([np.load(f)["true_start"] for f in files])
    pcas = {}
    for name, tok in (("DINO_refitPCA", dtok), ("IBOT_PCA", ibot)):
        t = torch.cat([tok(true_starts[i:i + 32]) for i in range(0, len(true_starts), 32)]).reshape(-1, 384)
        pcas[name] = fit_channel_pca(t)

    def encode(name, frames):
        if name == "DINO_fixedPCA":
            return lat(frames).reshape(len(frames), -1).astype(np.float64)
        tok = dtok if name.startswith("DINO") else ibot
        mean, basis, zm, zs = pcas[name]
        t = tok(frames)
        return (((t - mean) @ basis - zm) / zs).reshape(len(frames), -1).double().cpu().numpy()
    names = ("DINO_fixedPCA", "DINO_refitPCA", "IBOT_PCA")
    pos_cov, rot_cov = ps.pooled_error_model()
    per = {n: {"gain": [], "lin": [], "true": [], "alarm": []} for n in names}
    fork = []
    for path in files:
        fam, seed, s = path.stem.split("_")
        key = ({"F1": 1, "F2": 2}[fam], int(seed), int(s[1:]))
        with np.load(path) as f:
            starts, futures, true_start, harm = f["starts"], f["futures"], f["true_start"], f["harm"]
        n_obj = len(json.loads(json.dumps(ps.sample_scene(fam, int(seed))["objects"])))
        size = np.array([np.sqrt(np.mean(np.sum((ps.draw_error(n_obj, pos_cov, rot_cov, MONITOR_SEED, *key, j)[0] * 1000) ** 2, 1)))
                         for j in range(64)])
        k = int(harm.sum()); fork.append(2 <= k <= 62)
        for n in names:
            z0 = encode(n, true_start[None])[0]
            zs = encode(n, starts)
            d = np.linalg.norm(zs - z0, axis=1)
            per[n]["gain"].append(float(np.median(d / size)))
            per[n]["lin"].append(float(spearmanr(d, size).correlation))
            per[n]["true"].append(z0)
            fu = encode(n, futures.reshape(-1, 384, 384, 3)).reshape(64, 2, -1)
            per[n]["alarm"].append(alarm(fu[:, 0], fu[:, 1], "BOTH"))
        torch.cuda.empty_cache()
    fork = np.array(fork)
    out = {"states": len(files), "forks": int(fork.sum())}
    rng = np.random.default_rng(0)
    for n in names:
        T = np.stack(per[n]["true"])
        pairs = rng.integers(len(T), size=(2000, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        scale = float(np.median(np.linalg.norm(T[pairs[:, 0]] - T[pairs[:, 1]], axis=1)))
        out[n] = {"normalised_gain_per_mm": float(np.median(per[n]["gain"]) / scale), "scene_scale": scale,
                  "linearity_spearman": float(np.nanmedian(per[n]["lin"])),
                  "real_future_BOTH": score(np.array(per[n]["alarm"]), fork)}
    d, i = out["DINO_fixedPCA"], out["IBOT_PCA"]
    out["iBOT_qualifies"] = bool((i["normalised_gain_per_mm"] <= 0.5 * d["normalised_gain_per_mm"]
                                  or i["linearity_spearman"] >= d["linearity_spearman"] + 0.2)
                                 and i["real_future_BOTH"]["score"] >= d["real_future_BOTH"]["score"] - 0.05)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
