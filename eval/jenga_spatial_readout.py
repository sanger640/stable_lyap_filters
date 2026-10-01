"""Spatial (keypoint) readout of block pose from frozen patch-token grids.

Frozen in results/jenga/visual_encoder_precision_addendum_2.json. One identical head per encoder
and camera; 60 fixed epochs; held-out metrics from jenga_v1_probe.geometry_errors.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

import jenga_v1_probe as vp  # noqa: E402

ENCODED = ROOT / "results/program_b/vjepa"
POSE = 27
KEYPOINTS, WIDTH, EPOCHS, BATCH = 16, 128, 60, 64


def load(directory):
    grids, states, ids = [], [], []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        with np.load(path) as d:
            z = d["latents"][:, 0]                                  # (N, tokens, 384) float16
            side = int(math.isqrt(z.shape[1]))
            grids.append(z.reshape(len(z), side, side, -1))
            states.append(d["state"])
            ids.extend([f"{d['episode_id']}:{int(d['seed'])}"] * len(z))
    return np.concatenate(grids), np.concatenate(states), np.asarray(ids)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class KeypointHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels + 2, WIDTH, 1), nn.GELU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(3)])
        self.heat = nn.Conv2d(WIDTH, KEYPOINTS, 1)
        self.mlp = nn.Sequential(nn.Linear(KEYPOINTS * (2 + WIDTH), 512), nn.GELU(),
                                 nn.Linear(512, POSE))

    def forward(self, grid):                                        # (B, H, W, C)
        x = grid.permute(0, 3, 1, 2)
        b, _, h, w = x.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        coords = torch.stack([xs, ys])[None].expand(b, -1, -1, -1)
        f = self.blocks(self.stem(torch.cat([x, coords], 1)))
        attention = torch.softmax(self.heat(f).flatten(2), -1)     # (B, K, HW)
        xy = attention @ coords[0].flatten(1).T                    # (B, K, 2)
        pooled = attention @ f.flatten(2).transpose(1, 2)          # (B, K, WIDTH)
        return self.mlp(torch.cat([xy, pooled], -1).flatten(1))


def run(directory, device="cuda", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    grids, states, ids = load(directory)
    unique = sorted(set(ids)); val = set(unique[:max(1, int(len(unique) * 0.12))])
    held = np.array([i in val for i in ids])
    y = torch.from_numpy(states[:, :POSE])
    mean, std = y[~held].mean(0), y[~held].std(0).clamp_min(1e-6)
    x_train = torch.from_numpy(grids[~held]); y_train = (y[~held] - mean) / std
    model = KeypointHead(grids.shape[-1]).to(device)
    steps = EPOCHS * math.ceil(len(x_train) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x_train))
        total = 0.0
        for first in range(0, len(order), BATCH):
            idx = order[first:first + BATCH]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x_train[idx].to(device).float())
            loss = ((pred.float() - y_train[idx].to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}: train loss {total / len(x_train):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        x_held = torch.from_numpy(grids[held])
        for first in range(0, len(x_held), 256):
            preds.append(model(x_held[first:first + 256].to(device).float()).float().cpu())
    pose = torch.cat(preds) * std + mean
    full = torch.from_numpy(states[held]).clone()
    predicted = full.clone(); predicted[:, :POSE] = pose        # only pose is predicted
    g = vp.geometry_errors(predicted, full)
    return {k: g[k] for k in ("relative_position_mm", "common_mode_mm", "absolute_position_mm",
                              "gap_grasped_left_mm", "gap_grasped_right_mm", "gap_left_right_mm",
                              "orientation_deg")}, int(held.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "results/jenga/visual_spatial_readout.json"))
    args = ap.parse_args()
    results = {}
    for encoder in ("dinov2_392", "vjepa2_vitg_384", "vjepa2_1_vitg_384"):
        for camera, tag in (("default", ""), ("close", "_close")):
            print(f"{encoder} {camera}", flush=True)
            metrics, held = run(ENCODED / f"enc_{encoder}{tag}")
            results[f"{encoder}_{camera}"] = {"held_out_states": held, **metrics}
            print("  " + json.dumps({k: round(v, 3) for k, v in metrics.items()}), flush=True)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    best = min(results.values(), key=lambda r: r["relative_position_mm"])
    gaps = ("gap_grasped_left_mm", "gap_grasped_right_mm", "gap_left_right_mm")
    promising = any(r["relative_position_mm"] <= 1.0 and all(r[g] <= 1.0 for g in gaps)
                    for r in results.values())
    partial = best["relative_position_mm"] <= 0.7 * 4.08
    results["_decision"] = "promising" if promising else "partial" if partial else "negative"
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print("decision:", results["_decision"])


if __name__ == "__main__":
    main()
