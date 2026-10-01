"""Encode the cached V1 frames with a frozen visual encoder, in jenga_v1_encode's output format.

Protocol: results/jenga/visual_encoder_precision_protocol.json. Output npz per file: latents
(N, 1, tokens, 384) float16, state, starts, episode_id, seed, pixels -- consumed unchanged by
eval/jenga_v1_probe.py --encoded <dir> --frames 1.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external/vjepa2"))

FRAMES = ROOT / "results/program_b/vjepa/frames_384"
OUT = ROOT / "results/program_b/vjepa"
CHANNELS = 384
VAL_FRACTION = 0.12
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def training_files(files):
    """Same split as jenga_v1_probe: first 12% of sorted 'episode:seed' ids are validation."""
    ids = {}
    for f in files:
        with np.load(f) as d:
            ids[f] = f"{d['episode_id']}:{int(d['seed'])}"
    unique = sorted(set(ids.values()))
    val = set(unique[:max(1, int(len(unique) * VAL_FRACTION))])
    return [f for f in files if ids[f] not in val]


def load_vjepa(name, device):
    import src.hub.backbones as bb
    bb.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    encoder, _ = getattr(bb, name)(pretrained=True)
    return encoder.to(device).eval()   # fp32 weights; autocast at inference (2.1's RoPE upcasts q/k)


def vjepa_tokens(encoder, frames, device, batch=8):
    out = []
    for first in range(0, len(frames), batch):
        x = torch.as_tensor(frames[first:first + batch], device=device).permute(0, 3, 1, 2).float() / 255
        x = x[:, :, None].repeat(1, 1, 2, 1, 1)                     # (B, 3, T=2, H, W)
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(x).float())
    return torch.cat(out)                                           # (N, 576, 1408)


def dino_tokens(encoder, frames, device, batch=16):
    from torchvision.transforms import functional as TF
    out = []
    for first in range(0, len(frames), batch):
        x = torch.as_tensor(frames[first:first + batch], device=device).permute(0, 3, 1, 2).float() / 255
        x = (TF.resize(x, [392, 392], antialias=True) - 0.5) / 0.5
        with torch.inference_mode():
            out.append(encoder.forward(x).float())
    return torch.cat(out)                                           # (N, 784, 384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=("dinov2_392", "vjepa2_vitg_384", "vjepa2_1_vitg_384"))
    ap.add_argument("--frames-dir", default=str(FRAMES))
    ap.add_argument("--tag", default="", help="suffix for the output directory (e.g. _close)")
    ap.add_argument("--fit-files", type=int, default=60)
    ap.add_argument("--fit-tokens", type=int, default=200_000)
    args = ap.parse_args()
    device = "cuda"
    files = sorted(Path(args.frames_dir).glob("ep*_seed*.npz"))
    out = OUT / f"enc_{args.encoder}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    if args.encoder == "dinov2_392":
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        encoder = load_world_model(DEFAULT_CHECKPOINT, device).encoder
        tokens_of, basis, mean = (lambda f: dino_tokens(encoder, f, device)), None, None
    else:
        name = {"vjepa2_vitg_384": "vjepa2_vit_giant_384",
                "vjepa2_1_vitg_384": "vjepa2_1_vit_giant_384"}[args.encoder]
        encoder = load_vjepa(name, device)
        tokens_of = lambda f: vjepa_tokens(encoder, f, device)
        # Channel PCA on TRAINING tokens only.
        rng = np.random.default_rng(0)
        fit = rng.choice(training_files(files), args.fit_files, replace=False)
        sample = []
        for f in fit:
            with np.load(f) as d:
                t = tokens_of(d["frames"]).reshape(-1, 1408)
            sample.append(t[torch.randperm(len(t), device=device)[:args.fit_tokens // len(fit)]])
        sample = torch.cat(sample)
        mean = sample.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(sample - mean, q=CHANNELS, niter=4)
        basis = v[:, :CHANNELS]
        explained = float(((sample - mean) @ basis).var(0).sum() / (sample - mean).var(0).sum())
        print(f"channel PCA 1408 -> {CHANNELS}: {explained:.3f} of token variance", flush=True)
        torch.save({"mean": mean.cpu(), "basis": basis.cpu(), "explained": explained,
                    "fit_files": [Path(f).name for f in fit]}, out / "channel_pca.pt")
    for number, f in enumerate(files, 1):
        target = out / Path(f).name
        if target.exists():
            continue
        with np.load(f) as d:
            tokens = tokens_of(d["frames"])
            if basis is not None:
                tokens = (tokens - mean) @ basis
            np.savez(target, latents=tokens[:, None].half().cpu().numpy(), state=d["state"],
                     starts=d["starts"], episode_id=d["episode_id"], seed=d["seed"],
                     pixels=d["pixels"])
        if number % 50 == 0 or number == len(files):
            print(f"{number}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
