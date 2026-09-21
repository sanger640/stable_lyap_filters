"""V1: can DINO recover the 61-dim state the frozen dynamics model needs?

The pipeline under test is  DINO latent -> x_hat (61D) -> the already-trained W6 dynamics, with the
dynamics FROZEN. If fork recall falls, the bottleneck is perception or state estimation, not the
learned dynamics.

Two things are varied, because either could be the reason a probe fails:

  input   a SINGLE frame at the chunk start, or a SHORT HISTORY of the frames before it. A still
          image cannot distinguish a block at rest from the same block moving, so velocity (18 of
          the 61 numbers) may simply not be observable from one frame. The history variant is the
          fair test of whether vision contains the state at all.
  probe   a regularised LINEAR map, and a small MLP. If linear fails where the MLP succeeds, the
          information is present but not linearly decodable -- which is a statement about the
          decoder, not about DINO.

Latents are projected onto a PCA basis fitted on TRAINING configurations only; the probes are
fitted there too, and reported on held-out episodes here and on the batch-3 scenes by
`jenga_v1_gate.py`. Reconstruction error is reported per state group, in the units that group is
measured in, because a millimetre of block position and a contact flag are not comparable.
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from jenga_w5_train import start_from_full_state  # noqa: E402

GROUPS = {"position": (slice(0, 9), "mm", 1000.0), "rotation": (slice(9, 27), "unit", 1.0),
          "velocity": (slice(27, 45), "m/s or rad/s", 1.0),
          "contact": (slice(45, 57), "flag", 1.0), "gripper": (slice(57, 61), "mm", 1000.0)}


def load_pairs(bulk_dir, trace_dir, frames):
    """-> latents (N, frames, 196*384) float32, states (N, 61), ids (N,)."""
    latents, states, ids = [], [], []
    for path in sorted(glob.glob(str(Path(bulk_dir) / "ep*_seed*.npz"))):
        name = Path(path).name
        trace = Path(trace_dir) / name
        if not trace.exists():
            continue
        b = np.load(path, allow_pickle=False)
        t = np.load(trace, allow_pickle=False)
        if not np.array_equal(b["starts"], t["starts"]):
            raise RuntimeError(f"{name}: bulk and trace chunk starts differ")
        history = b["history_latents"][:, -frames:]            # (S, frames, 196, 384)
        latents.append(history.reshape(len(history), frames, -1).astype(np.float32))
        states.append(start_from_full_state(t["start_state"]))
        ids.extend([f"{b['episode_id']}:{b['seed']}"] * len(history))
    return np.concatenate(latents), np.concatenate(states), np.asarray(ids)


def fit_basis(latents, components, device):
    """PCA basis over the flattened tokens of TRAINING rows, fitted once and reused everywhere."""
    flat = torch.as_tensor(latents.reshape(-1, latents.shape[-1]), device=device)
    mean = flat.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(flat - mean, q=min(components, min(flat.shape) - 1), niter=4)
    return mean, v


def project(latents, mean, basis, device, batch=512):
    out = []
    for first in range(0, len(latents), batch):
        x = torch.as_tensor(latents[first:first + batch], device=device)
        out.append(((x - mean) @ basis).reshape(len(x), -1).cpu())
    return torch.cat(out)


def fit_linear(x, y, ridge):
    """Closed-form ridge with an intercept, solved in float64."""
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1).double()
    y = y.double()
    gram = x.T @ x
    gram += ridge * torch.eye(len(gram), dtype=gram.dtype) * gram.diagonal().mean()
    return torch.linalg.solve(gram, x.T @ y).float()


def apply_linear(weights, x):
    return torch.cat([x, torch.ones(len(x), 1)], 1) @ weights


def fit_mlp(x, y, device, hidden=512, epochs=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(),
                                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                torch.nn.Linear(hidden, y.shape[1])).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = x.to(device), y.to(device)
    for epoch in range(epochs):
        order = torch.randperm(len(x), device=device)
        for first in range(0, len(x), 1024):
            rows = order[first:first + 1024]
            loss = torch.nn.functional.mse_loss(model(x[rows]), y[rows])
            optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    model.eval()
    return model


def errors(predicted, truth):
    out = {}
    for name, (where, unit, scale) in GROUPS.items():
        diff = (predicted[:, where] - truth[:, where]) * scale
        out[name] = {"rmse": float(diff.pow(2).mean().sqrt()), "unit": unit,
                     "target_sd": float((truth[:, where] * scale).std())}
        out[name]["relative"] = out[name]["rmse"] / max(out[name]["target_sd"], 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", default=str(ROOT / "results/jenga/bulk_data"))
    ap.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    ap.add_argument("--frames", type=int, default=3, help="1 = single frame, >1 = short history")
    ap.add_argument("--components", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save-probe", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latents, states, ids = load_pairs(args.bulk, args.trace, args.frames)
    unique = sorted(set(ids))
    val_ids = set(unique[:max(1, int(len(unique) * args.val_fraction))])
    is_val = np.array([i in val_ids for i in ids])
    print(f"{len(latents)} states, {args.frames} frame(s), "
          f"{int((~is_val).sum())} train / {int(is_val.sum())} held-out episodes", flush=True)

    # Basis and probes from TRAINING configurations only.
    mean, basis = fit_basis(latents[~is_val], args.components, device)
    features = project(latents, mean, basis, device)
    target = torch.from_numpy(states)
    train, held = features[~is_val], features[is_val]
    y_train, y_held = target[~is_val], target[is_val]

    result = {"frames": args.frames, "components": args.components,
              "states": len(latents), "held_out_states": int(is_val.sum()), "probes": {}}

    weights = fit_linear(train, y_train, args.ridge)
    result["probes"]["linear"] = {"train": errors(apply_linear(weights, train), y_train),
                                  "held_out": errors(apply_linear(weights, held), y_held)}
    mlp = fit_mlp(train, y_train, device)
    with torch.no_grad():
        result["probes"]["mlp"] = {
            "train": errors(mlp(train.to(device)).cpu(), y_train),
            "held_out": errors(mlp(held.to(device)).cpu(), y_held)}

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    if args.save_probe:
        torch.save({"mean": mean.cpu(), "basis": basis.cpu(), "linear": weights,
                    "mlp": mlp.state_dict(), "frames": args.frames,
                    "components": args.components}, args.save_probe)
    for probe in ("linear", "mlp"):
        line = " ".join(f"{g} {result['probes'][probe]['held_out'][g]['rmse']:.3f}"
                        f"({result['probes'][probe]['held_out'][g]['relative']:.2f})"
                        for g in GROUPS)
        print(f"  {probe:6s} held-out rmse(relative): {line}", flush=True)


if __name__ == "__main__":
    main()
