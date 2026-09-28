"""Prospectively frozen label-free training for the Panda slot-dynamics representation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from models.slot_dynamics import SlotDynamicsRepresentation
import panda_push_slot_data as data_source

OUT = data_source.OUT
PROTOCOL_MANIFEST = OUT / "training_protocol.json"
CHECKPOINT = OUT / "training_checkpoint.pt"
REPORT = OUT / "training_report.json"
MODEL_MANIFEST = OUT / "model_manifest.json"
SEED = 642
EPOCHS = 12
BATCH_PAIRS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MODEL_ARGS = {"input_dim": data_source.TOKEN_DIM, "slot_dim": 48, "slots": 8,
              "iterations": 3, "action_dim": 4}
LOSS_WEIGHTS = {"reconstruction": 1.0, "dynamics": 1.0, "geometry": 0.5,
                "pair_separation": 1.0, "mask_sharpness": 0.01,
                "slot_balance": 0.05}

PROTOCOL = {
    "id": "panda-anonymous-slot-dynamics-v0",
    "status": "label-free representation TRAIN/validation; monitor DEV unopened during training",
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_pairs": BATCH_PAIRS,
    "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                  "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0},
    "model": MODEL_ARGS,
    "loss_weights": LOSS_WEIGHTS,
    "losses": {
        "reconstruction": "reconstruct fixed projected DINO patch descriptors",
        "dynamics": "predict next anonymous slot from current slot and executed action",
        "geometry": "predicted slots reproduce next mask centroid/covariance/mass",
        "pair_separation": "predicted neighboring-action slot separation matches observed separation",
        "mask_sharpness": "low per-pixel slot assignment entropy",
        "slot_balance": "discourage globally dead slots",
    },
    "checkpoint_selection": "minimum complete validation objective; no task or physical labels",
    "monitor_interface": "slot x/y, finite-difference vx/vy, covariance and mass; PCA45 per state",
    "excluded": data_source.PROTOCOL["excluded"],
    "resource_policy": "CPU-compatible; Torch <=8 threads; no simulator during training",
}

FROZEN_FILES = ("src/models/slot_dynamics.py", "eval/panda_push_slot_data.py",
                "eval/panda_push_slot_train.py",
                "results/panda_block_push/slot_dynamics/data_manifest.json",
                "results/panda_block_push/slot_dynamics/unlabeled_tokens.npz")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_digest(files):
    payload = json.dumps({"protocol": PROTOCOL, "files_sha256": files},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze_protocol():
    if PROTOCOL_MANIFEST.exists():
        raise SystemExit(f"refusing to overwrite {PROTOCOL_MANIFEST}")
    data_source.verify()
    files = {name: sha256(ROOT / name) for name in FROZEN_FILES}
    value = {"protocol": PROTOCOL,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files, "protocol_sha256": _protocol_digest(files)}
    PROTOCOL_MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    print(value["protocol_sha256"])
    return value


def verify_protocol():
    data_source.verify()
    value = json.loads(PROTOCOL_MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL or tuple(value.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("slot training protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise SystemExit(f"slot training frozen file changed: {name}")
    if _protocol_digest(value["files_sha256"]) != value["protocol_sha256"]:
        raise SystemExit("slot training protocol digest mismatch")
    return value


def _normalization(descriptors, actions, train_rows):
    values = descriptors[train_rows].astype(np.float32)
    feature_mean = values.mean(axis=(0, 1, 2))
    feature_std = values.std(axis=(0, 1, 2)).clip(1e-4)
    selected_actions = actions[train_rows].astype(np.float32)
    action_mean = selected_actions.mean(axis=(0, 1))
    action_std = selected_actions.std(axis=(0, 1)).clip(1e-4)
    return feature_mean, feature_std, action_mean, action_std


def _pair_rows(pair_ids, selected_pairs):
    return np.stack([np.flatnonzero(pair_ids == pair) for pair in selected_pairs])


def objective(model, descriptors, actions, weights):
    """Complete label-free objective for ``(pairs,2,time,patch,feature)`` batches."""
    pairs, branches, time_steps, patches, dims = descriptors.shape
    flat = descriptors.reshape(-1, patches, dims)
    encoded = model.encode(flat)
    slots = encoded["slots"].reshape(pairs, branches, time_steps, model.slots, model.slot_dim)
    geometry = encoded["geometry"].reshape(pairs, branches, time_steps, model.slots, 6)
    reconstruction = encoded["reconstruction"].reshape_as(descriptors)
    masks = encoded["masks"]
    current = slots[:, :, :-1].reshape(-1, model.slots, model.slot_dim)
    action = actions.reshape(-1, actions.shape[-1])
    predicted = model.predict(current, action).reshape_as(slots[:, :, 1:])
    predicted_flat = predicted.reshape(-1, model.slots, model.slot_dim)
    _, predicted_masks = model.decode(predicted_flat, patches)
    predicted_geometry = model.geometry(predicted_masks).reshape_as(geometry[:, :, 1:])

    terms = {}
    terms["reconstruction"] = F.mse_loss(reconstruction, descriptors)
    terms["dynamics"] = F.smooth_l1_loss(predicted, slots[:, :, 1:].detach())
    terms["geometry"] = F.smooth_l1_loss(predicted_geometry, geometry[:, :, 1:].detach())
    predicted_separation = predicted[:, 0] - predicted[:, 1]
    observed_separation = (slots[:, 0, 1:] - slots[:, 1, 1:]).detach()
    terms["pair_separation"] = F.smooth_l1_loss(predicted_separation, observed_separation)
    terms["mask_sharpness"] = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log()).sum(1).mean()
    mass = encoded["geometry"][..., -1]
    terms["slot_balance"] = (mass.mean(0) - 1. / model.slots).square().mean()
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


def train(torch_threads=8):
    manifest = verify_protocol()
    if CHECKPOINT.exists() or REPORT.exists() or MODEL_MANIFEST.exists():
        raise SystemExit("refusing to overwrite an existing slot model result")
    torch.set_num_threads(torch_threads); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cache = np.load(data_source.TOKENS, allow_pickle=False)
    descriptors = cache["descriptors"]
    actions = cache["actions"].astype(np.float32)
    pair_ids = cache["pair_index"].astype(int); split = cache["split"].astype(int)
    train_pairs = np.unique(pair_ids[split == 0]); validation_pairs = np.unique(pair_ids[split == 1])
    train_rows = np.flatnonzero(split == 0)
    feature_mean, feature_std, action_mean, action_std = _normalization(
        descriptors, actions, train_rows)

    model = SlotDynamicsRepresentation(**MODEL_ARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED); history = []; best = float("inf"); best_state = None

    def batch(pair_batch):
        rows = _pair_rows(pair_ids, pair_batch)
        x = descriptors[rows].astype(np.float32)
        x = (x - feature_mean) / feature_std
        a = (actions[rows].astype(np.float32) - action_mean) / action_std
        return torch.from_numpy(x), torch.from_numpy(a)

    def run(selected, training):
        order = np.asarray(selected).copy()
        if training: rng.shuffle(order)
        aggregate = []
        model.train(training)
        for first in range(0, len(order), BATCH_PAIRS):
            x, a = batch(order[first:first + BATCH_PAIRS])
            with torch.set_grad_enabled(training):
                loss, terms = objective(model, x, a, LOSS_WEIGHTS)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            aggregate.append([float(loss.detach())] + [float(terms[name].detach())
                                                       for name in LOSS_WEIGHTS])
        mean = np.mean(aggregate, axis=0)
        return {"total": float(mean[0]),
                **{name: float(value) for name, value in zip(LOSS_WEIGHTS, mean[1:])}}

    for epoch in range(EPOCHS):
        start = time.time(); train_metrics = run(train_pairs, True)
        validation_metrics = run(validation_pairs, False)
        history.append({"epoch": epoch + 1, "train": train_metrics,
                        "validation": validation_metrics})
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{EPOCHS} train={train_metrics['total']:.4f} "
              f"validation={validation_metrics['total']:.4f} ({time.time()-start:.1f}s)", flush=True)

    checkpoint = {"model_args": MODEL_ARGS, "model": best_state,
                  "feature_mean": feature_mean, "feature_std": feature_std,
                  "action_mean": action_mean, "action_std": action_std,
                  "protocol_sha256": manifest["protocol_sha256"]}
    torch.save(checkpoint, CHECKPOINT)
    best_epoch = int(np.argmin([row["validation"]["total"] for row in history])) + 1
    report = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "best_epoch": best_epoch, "best_validation_total": best, "history": history}
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    model_manifest = {"protocol_sha256": manifest["protocol_sha256"],
                      "checkpoint_sha256": sha256(CHECKPOINT), "report_sha256": sha256(REPORT)}
    MODEL_MANIFEST.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_total": best,
                      "checkpoint_sha256": model_manifest["checkpoint_sha256"]}, indent=2))
    return report


def verify_model():
    protocol = verify_protocol(); value = json.loads(MODEL_MANIFEST.read_text())
    if value.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise SystemExit("slot model protocol mismatch")
    if sha256(CHECKPOINT) != value["checkpoint_sha256"] or sha256(REPORT) != value["report_sha256"]:
        raise SystemExit("slot model artifacts changed")
    return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("verify"); sub.add_parser("verify-model")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4, 6, 8), default=8)
    args = parser.parse_args()
    if args.command == "freeze": freeze_protocol()
    elif args.command == "verify": print(verify_protocol()["protocol_sha256"])
    elif args.command == "verify-model": print(verify_model()["checkpoint_sha256"])
    else: train(args.torch_threads)


if __name__ == "__main__":
    main()
