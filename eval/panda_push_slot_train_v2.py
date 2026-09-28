"""Balanced motion-slot v2 and its prospective label-free pre-monitor quality gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from models.balanced_motion_slots import BalancedMotionSlotRepresentation
import panda_push_slot_data as data_source
import panda_push_slot_train_v1 as base

OUT = data_source.OUT
PROTOCOL_MANIFEST = OUT / "training_protocol_v2.json"
CHECKPOINT = OUT / "training_checkpoint_v2.pt"
REPORT = OUT / "training_report_v2.json"
MODEL_MANIFEST = OUT / "model_manifest_v2.json"
QUALITY = OUT / "quality_report_v2.json"
QUALITY_MANIFEST = OUT / "quality_manifest_v2.json"
SEED, EPOCHS, BATCH_PAIRS = 644, 12, 4
MODEL_ARGS = {"descriptor_dim": data_source.TOKEN_DIM, "slot_dim": 48, "slots": 8,
              "iterations": 3, "action_dim": 4}
LOSS_WEIGHTS = {"motion_reconstruction": 1.0, "dynamics": 1.0, "geometry": 0.5,
                "pair_separation": 1.0, "motion_assignment": 0.5,
                "mask_sharpness": 0.05, "slot_balance": 0.20}
QUALITY_GATES = {"minimum_effective_slots": 6.0, "maximum_slot_mass": 0.25,
                 "minimum_geometry_motion_rms": 0.002,
                 "minimum_motion_map_correlation": 0.20}
PROTOCOL = {
    "id": "panda-balanced-motion-slot-dynamics-v2",
    "motivation": "v1 rejected pre-monitor because one slot monopolized all patches",
    "seed": SEED, "epochs": EPOCHS, "batch_pairs": BATCH_PAIRS,
    "model": MODEL_ARGS, "loss_weights": LOSS_WEIGHTS,
    "assignment": "six-iteration differentiable balanced slot/patch normalization",
    "checkpoint_selection": "minimum complete label-free validation objective",
    "pre_monitor_quality_gates": QUALITY_GATES,
    "quality_information": "validation descriptors and actions only; no task/physical labels",
    "excluded": data_source.PROTOCOL["excluded"],
    "resource_policy": "CPU-compatible; Torch <=8 threads; no simulator during training",
}
FROZEN_FILES = ("src/models/slot_dynamics.py", "src/models/motion_slot_dynamics.py",
                "src/models/balanced_motion_slots.py", "eval/panda_push_slot_data.py",
                "eval/panda_push_slot_train_v1.py", "eval/panda_push_slot_train_v2.py",
                "results/panda_block_push/slot_dynamics/data_manifest.json",
                "results/panda_block_push/slot_dynamics/unlabeled_tokens.npz")


def _configure_base():
    base.PROTOCOL_MANIFEST = PROTOCOL_MANIFEST; base.CHECKPOINT = CHECKPOINT
    base.REPORT = REPORT; base.MODEL_MANIFEST = MODEL_MANIFEST
    base.SEED = SEED; base.EPOCHS = EPOCHS; base.BATCH_PAIRS = BATCH_PAIRS
    base.MODEL_ARGS = MODEL_ARGS; base.LOSS_WEIGHTS = LOSS_WEIGHTS
    base.PROTOCOL = PROTOCOL; base.FROZEN_FILES = FROZEN_FILES
    base.MotionSlotDynamicsRepresentation = BalancedMotionSlotRepresentation


def freeze_protocol():
    _configure_base(); return base.freeze_protocol()


def verify_protocol():
    _configure_base(); return base.verify_protocol()


def _correlation(left, right):
    x = np.asarray(left, float).reshape(-1); y = np.asarray(right, float).reshape(-1)
    x -= x.mean(); y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else 0.0


def quality_report():
    _configure_base(); base.verify_model()
    if QUALITY.exists() or QUALITY_MANIFEST.exists():
        raise SystemExit("refusing to overwrite slot v2 quality result")
    cache = np.load(data_source.TOKENS, allow_pickle=False)
    validation = np.flatnonzero(cache["split"] == 1)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = BalancedMotionSlotRepresentation(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"]); model.eval()
    descriptor = cache["descriptors"][validation].astype(np.float32)
    descriptor = (descriptor - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    masks, geometry, slots = [], [], []
    with torch.inference_mode():
        for first in range(0, len(descriptor), 8):
            current = torch.from_numpy(descriptor[first:first + 8])
            start = current[:, :1].expand_as(current)
            batch, time, patches, dims = current.shape
            out = model.encode(current.reshape(-1, patches, dims),
                               start.reshape(-1, patches, dims))
            masks.append(out["masks"].reshape(batch, time, model.slots, patches).numpy())
            geometry.append(out["geometry"].reshape(batch, time, model.slots, 6).numpy())
            slots.append(out["slots"].reshape(batch, time, model.slots, model.slot_dim).numpy())
    masks = np.concatenate(masks); geometry = np.concatenate(geometry); slots = np.concatenate(slots)
    mass = masks.mean(-1)
    effective = np.exp(-np.sum(mass * np.log(np.maximum(mass, 1e-12)), axis=-1))
    geometry_motion = float(np.sqrt(np.mean(np.diff(geometry, axis=1) ** 2)))
    slot_motion = np.linalg.norm(slots - slots[:, :1], axis=-1)
    predicted_motion = np.sum(masks * slot_motion[..., None], axis=2)
    observed_motion = np.linalg.norm(descriptor - descriptor[:, :1], axis=-1)
    motion_correlation = _correlation(predicted_motion[:, 1:], observed_motion[:, 1:])
    metrics = {"mean_effective_slots": float(effective.mean()),
               "maximum_slot_mass": float(mass.max()),
               "geometry_motion_rms": geometry_motion,
               "motion_map_correlation": motion_correlation}
    gates = {"effective_slots_pass": metrics["mean_effective_slots"] >=
                                      QUALITY_GATES["minimum_effective_slots"],
             "slot_mass_pass": metrics["maximum_slot_mass"] <=
                               QUALITY_GATES["maximum_slot_mass"],
             "geometry_motion_pass": geometry_motion >=
                                     QUALITY_GATES["minimum_geometry_motion_rms"],
             "motion_correlation_pass": motion_correlation >=
                                        QUALITY_GATES["minimum_motion_map_correlation"]}
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "metrics": metrics, "gates": gates, "passed": bool(all(gates.values()))}
    QUALITY.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    manifest = {"checkpoint_sha256": base.sha256(CHECKPOINT),
                "quality_sha256": base.sha256(QUALITY)}
    QUALITY_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(value, indent=2)); return value


def train(torch_threads=8):
    _configure_base(); report = base.train(torch_threads)
    quality_report(); return report


def verify_model():
    _configure_base(); model_manifest = base.verify_model()
    quality_manifest = json.loads(QUALITY_MANIFEST.read_text())
    if quality_manifest["checkpoint_sha256"] != base.sha256(CHECKPOINT) or \
            quality_manifest["quality_sha256"] != base.sha256(QUALITY):
        raise SystemExit("slot v2 quality artifacts changed")
    return model_manifest, json.loads(QUALITY.read_text())


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze"); sub.add_parser("verify"); sub.add_parser("verify-model")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--torch-threads", type=int, choices=(1,2,4,6,8), default=8)
    args = parser.parse_args()
    if args.command == "freeze": freeze_protocol()
    elif args.command == "verify": print(verify_protocol()["protocol_sha256"])
    elif args.command == "verify-model": print(verify_model()[1]["passed"])
    else: train(args.torch_threads)


if __name__ == "__main__": main()
