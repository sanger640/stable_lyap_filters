"""Motion-conditioned label-free slot training, prospectively frozen after v0 collapse audit."""
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

from models.motion_slot_dynamics import MotionSlotDynamicsRepresentation
import panda_push_slot_data as data_source
import panda_push_slot_train as v0

OUT = data_source.OUT
PROTOCOL_MANIFEST = OUT / "training_protocol_v1.json"
CHECKPOINT = OUT / "training_checkpoint_v1.pt"
REPORT = OUT / "training_report_v1.json"
MODEL_MANIFEST = OUT / "model_manifest_v1.json"
SEED, EPOCHS, BATCH_PAIRS = 643, 12, 4
LEARNING_RATE, WEIGHT_DECAY = 3e-4, 1e-5
MODEL_ARGS = {"descriptor_dim": data_source.TOKEN_DIM, "slot_dim": 48, "slots": 8,
              "iterations": 3, "action_dim": 4}
LOSS_WEIGHTS = {"motion_reconstruction": 1.0, "dynamics": 1.0, "geometry": 0.5,
                "pair_separation": 1.0, "motion_assignment": 0.5,
                "mask_sharpness": 0.10, "slot_balance": 0.20}
PROTOCOL = {
    "id": "panda-motion-conditioned-anonymous-slot-dynamics-v1",
    "motivation": "v0 rejected before monitor scoring: validation mask entropy remained near ln(8)",
    "seed": SEED, "epochs": EPOCHS, "batch_pairs": BATCH_PAIRS,
    "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE,
                  "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0},
    "model": MODEL_ARGS, "loss_weights": LOSS_WEIGHTS,
    "input": "current projected DINO descriptor plus difference from common pre-action descriptor",
    "reconstruction": "sample-specific descriptor motion only, not static appearance",
    "motion_assignment": "slot motion magnitude reconstructs per-patch descriptor-change magnitude",
    "checkpoint_selection": "minimum complete label-free validation objective",
    "pre_monitor_gate": {"validation_mask_entropy_below": 1.8,
                         "validation_motion_reconstruction_below_epoch1": True},
    "excluded": data_source.PROTOCOL["excluded"],
    "resource_policy": "CPU-compatible; Torch <=8 threads; no simulator during training",
}
FROZEN_FILES = ("src/models/slot_dynamics.py", "src/models/motion_slot_dynamics.py",
                "eval/panda_push_slot_data.py", "eval/panda_push_slot_train_v1.py",
                "results/panda_block_push/slot_dynamics/data_manifest.json",
                "results/panda_block_push/slot_dynamics/unlabeled_tokens.npz")


def sha256(path): return v0.sha256(path)


def _digest(files):
    value = json.dumps({"protocol": PROTOCOL, "files_sha256": files},
                       sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def freeze_protocol():
    if PROTOCOL_MANIFEST.exists(): raise SystemExit(f"refusing to overwrite {PROTOCOL_MANIFEST}")
    data_source.verify(); files = {name: sha256(ROOT / name) for name in FROZEN_FILES}
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files, "protocol_sha256": _digest(files)}
    PROTOCOL_MANIFEST.write_text(json.dumps(value, indent=2) + "\n")
    print(value["protocol_sha256"]); return value


def verify_protocol():
    data_source.verify(); value = json.loads(PROTOCOL_MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL or tuple(value.get("files_sha256", {})) != FROZEN_FILES:
        raise SystemExit("slot v1 protocol mismatch")
    for name, expected in value["files_sha256"].items():
        if sha256(ROOT / name) != expected: raise SystemExit(f"slot v1 frozen file changed: {name}")
    if _digest(value["files_sha256"]) != value["protocol_sha256"]:
        raise SystemExit("slot v1 protocol digest mismatch")
    return value


def objective(model, descriptors, actions, weights):
    pairs, branches, time_steps, patches, dims = descriptors.shape
    start = descriptors[:, :, :1].expand_as(descriptors)
    current_flat = descriptors.reshape(-1, patches, dims)
    start_flat = start.reshape(-1, patches, dims)
    encoded = model.encode(current_flat, start_flat)
    slots = encoded["slots"].reshape(pairs, branches, time_steps, model.slots, model.slot_dim)
    geometry = encoded["geometry"].reshape(pairs, branches, time_steps, model.slots, 6)
    reconstruction = encoded["reconstruction"].reshape_as(descriptors)
    target_delta = descriptors - start
    masks = encoded["masks"].reshape(pairs, branches, time_steps, model.slots, patches)
    current = slots[:, :, :-1].reshape(-1, model.slots, model.slot_dim)
    predicted = model.predict(current, actions.reshape(-1, actions.shape[-1])).reshape_as(
        slots[:, :, 1:])
    predicted_flat = predicted.reshape(-1, model.slots, model.slot_dim)
    _, predicted_masks = model.decode(predicted_flat, patches)
    predicted_geometry = model.geometry(predicted_masks).reshape_as(geometry[:, :, 1:])

    terms = {}
    terms["motion_reconstruction"] = F.mse_loss(reconstruction, target_delta)
    terms["dynamics"] = F.smooth_l1_loss(predicted, slots[:, :, 1:].detach())
    terms["geometry"] = F.smooth_l1_loss(predicted_geometry, geometry[:, :, 1:].detach())
    terms["pair_separation"] = F.smooth_l1_loss(
        predicted[:, 0] - predicted[:, 1],
        (slots[:, 0, 1:] - slots[:, 1, 1:]).detach())
    slot_motion = torch.linalg.norm(slots - slots[:, :, :1], dim=-1)
    predicted_motion = torch.sum(masks * slot_motion[..., None], dim=3)
    observed_motion = torch.linalg.norm(target_delta, dim=-1)
    predicted_motion = predicted_motion / predicted_motion.mean(-1, keepdim=True).clamp_min(1e-4)
    observed_motion = observed_motion / observed_motion.mean(-1, keepdim=True).clamp_min(1e-4)
    # The common start has zero motion in both tensors and carries no assignment information.
    terms["motion_assignment"] = F.smooth_l1_loss(predicted_motion[:, :, 1:],
                                                    observed_motion[:, :, 1:])
    flat_masks = masks.reshape(-1, model.slots, patches)
    terms["mask_sharpness"] = -(flat_masks.clamp_min(1e-8) *
                                 flat_masks.clamp_min(1e-8).log()).sum(1).mean()
    mass = flat_masks.mean(-1)
    terms["slot_balance"] = (mass.mean(0) - 1. / model.slots).square().mean()
    return sum(weights[name] * value for name, value in terms.items()), terms


def train(torch_threads=8):
    manifest = verify_protocol()
    if any(path.exists() for path in (CHECKPOINT, REPORT, MODEL_MANIFEST)):
        raise SystemExit("refusing to overwrite an existing slot v1 result")
    torch.set_num_threads(torch_threads); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cache = np.load(data_source.TOKENS, allow_pickle=False)
    descriptors = cache["descriptors"]; actions = cache["actions"].astype(np.float32)
    pair_ids = cache["pair_index"].astype(int); split = cache["split"].astype(int)
    train_pairs = np.unique(pair_ids[split == 0]); validation_pairs = np.unique(pair_ids[split == 1])
    train_rows = np.flatnonzero(split == 0)
    feature_mean, feature_std, action_mean, action_std = v0._normalization(
        descriptors, actions, train_rows)
    model = MotionSlotDynamicsRepresentation(**MODEL_ARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED); history = []; best = float("inf"); best_state = None

    def batch(pair_batch):
        rows = v0._pair_rows(pair_ids, pair_batch)
        x = (descriptors[rows].astype(np.float32) - feature_mean) / feature_std
        a = (actions[rows].astype(np.float32) - action_mean) / action_std
        return torch.from_numpy(x), torch.from_numpy(a)

    def run(selected, training):
        order = np.asarray(selected).copy()
        if training: rng.shuffle(order)
        values = []; model.train(training)
        for first in range(0, len(order), BATCH_PAIRS):
            x, a = batch(order[first:first + BATCH_PAIRS])
            with torch.set_grad_enabled(training): loss, terms = objective(model, x, a, LOSS_WEIGHTS)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            values.append([float(loss.detach())] + [float(terms[name].detach()) for name in LOSS_WEIGHTS])
        mean = np.mean(values, axis=0)
        return {"total": float(mean[0]), **{name: float(value)
                for name, value in zip(LOSS_WEIGHTS, mean[1:])}}

    for epoch in range(EPOCHS):
        start_time = time.time(); train_metrics = run(train_pairs, True)
        validation_metrics = run(validation_pairs, False)
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"v1 epoch {epoch+1}/{EPOCHS} train={train_metrics['total']:.4f} "
              f"validation={validation_metrics['total']:.4f} "
              f"entropy={validation_metrics['mask_sharpness']:.3f} ({time.time()-start_time:.1f}s)",
              flush=True)
    best_epoch = int(np.argmin([row["validation"]["total"] for row in history])) + 1
    selected = history[best_epoch - 1]["validation"]
    gate = {"mask_entropy": selected["mask_sharpness"],
            "mask_entropy_pass": selected["mask_sharpness"] < 1.8,
            "motion_reconstruction_pass": selected["motion_reconstruction"] <
                                           history[0]["validation"]["motion_reconstruction"]}
    checkpoint = {"model_args": MODEL_ARGS, "model": best_state,
                  "feature_mean": feature_mean, "feature_std": feature_std,
                  "action_mean": action_mean, "action_std": action_std,
                  "protocol_sha256": manifest["protocol_sha256"]}
    torch.save(checkpoint, CHECKPOINT)
    report = {"protocol": PROTOCOL, "protocol_sha256": manifest["protocol_sha256"],
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch,
              "best_validation_total": best, "pre_monitor_gate": gate, "history": history}
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    model_manifest = {"protocol_sha256": manifest["protocol_sha256"],
                      "checkpoint_sha256": sha256(CHECKPOINT), "report_sha256": sha256(REPORT)}
    MODEL_MANIFEST.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "validation": selected,
                      "pre_monitor_gate": gate,
                      "checkpoint_sha256": model_manifest["checkpoint_sha256"]}, indent=2))
    return report


def verify_model():
    protocol = verify_protocol(); value = json.loads(MODEL_MANIFEST.read_text())
    if value.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise SystemExit("slot v1 model protocol mismatch")
    if sha256(CHECKPOINT) != value["checkpoint_sha256"] or sha256(REPORT) != value["report_sha256"]:
        raise SystemExit("slot v1 artifacts changed")
    return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("verify"); sub.add_parser("verify-model")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--torch-threads", type=int, choices=(1,2,4,6,8), default=8)
    args = parser.parse_args()
    if args.command == "freeze": freeze_protocol()
    elif args.command == "verify": print(verify_protocol()["protocol_sha256"])
    elif args.command == "verify-model": print(verify_model()["checkpoint_sha256"])
    else: train(args.torch_threads)


if __name__ == "__main__": main()
