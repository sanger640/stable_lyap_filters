"""Action-conditioned smooth-versus-branch monitor on the frozen D2 Jenga benchmark.

The runtime rule sees only the intended/noisy action windows and D2's predicted trajectories.
Physical fork/quiet classes are attached after the alarm for evaluation and never enter fitting,
candidate selection or model evidence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import result_dict, smooth_vs_branch_alarm  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, checkpoint_identity, verify  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT  # noqa: E402
from jenga_w5_eval import hold_index, load_model, predict  # noqa: E402


def pose_features(pose):
    """Dimensionless block pose shared by simulator and D2 (positions then rotation-6D)."""
    pose = np.asarray(pose, float).copy()
    pose[..., :9] /= 0.05  # Jenga block scale, identical to state_dynamics.POSITION_SCALE_M
    return pose


def sampled_predicted_pose(predicted, model):
    """Sample D2 at exactly the five hold times present in the simulator cache."""
    indices = [hold_index(model, held) for held in RECORD_AT]
    return pose_features(np.asarray(predicted)[:, indices, :27])


def action_errors(windows):
    """The H=8 xyz target perturbations; history, common hold and gripper are invariant."""
    chunk = np.asarray(windows, float)[:, 2:10, :3]
    return chunk - chunk.mean(0, keepdims=True)


def evaluate(path, selected_scales):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(path, device)
    rows = []
    for split in ("dev", "test"):
        for i, start in enumerate(bench[f"{split}_start"]):
            for k, value in enumerate(EVAL_CONFIG["scales"]):
                if value not in selected_scales:
                    continue
                windows = bench[f"{split}_windows"][i, k]
                predicted = predict(model, scale, start, windows, device)
                features = sampled_predicted_pose(predicted, model)
                early_stop = RECORD_AT.index(10) + 1
                result = smooth_vs_branch_alarm(
                    action_errors(windows), features[:, :early_stop], features)
                rows.append({
                    "split": split,
                    "episode_id": str(bench[f"{split}_episode"][i]),
                    "chunk_start": int(bench[f"{split}_chunk"][i]),
                    "scale": str(value),
                    "class": str(bench[f"{split}_classes"][i, k]),
                    "topple_count": int(bench[f"{split}_topples"][i, k]),
                    **result_dict(result),
                })
    return rows


def summarise(rows, scales):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    out = {}
    for split in ("dev", "test"):
        out[split] = {}
        for scale in map(str, scales):
            part = [r for r in rows if r["split"] == split and r["scale"] == scale]
            out[split][scale] = {}
            for cls in classes:
                selected = [r for r in part if r["class"] == cls]
                entry = {
                    "states": len(selected), "alarms": sum(r["alarm"] for r in selected),
                    "rate": float(np.mean([r["alarm"] for r in selected])) if selected else None,
                }
                if all("action_connected" in r for r in selected):
                    entry["action_disconnected"] = sum(
                        not r["action_connected"] for r in selected)
                out[split][scale][cls] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0],
                    choices=EVAL_CONFIG["scales"])
    args = ap.parse_args()
    rows = evaluate(args.model, args.scales)
    result = {
        "protocol": {
            "calibration_free": True,
            "input": "H=8 xyz execution perturbations and D2 block-pose trajectories sampled "
                     "at simulator hold steps 5,10,20,29,30; robot motion conditions prediction "
                     "but is not scored",
            "smooth": "one RBF-GP action-to-trajectory response surface",
            "branch": "two RBF-GP surfaces on action regions; fragmented boundaries pay a "
                      "graph description-length charge",
            "decision": "branch evidence minus smooth evidence and split-search description length "
                        "is positive at both hold 10 and hold 30",
            "candidate": "best residual-PC1 or trajectory-PC1 split; candidates charged in MDL",
            "labels": "physical classes are grading only",
            "uses_reference_states": False,
            "scales": args.scales,
        },
        "model": checkpoint_identity(args.model),
        "summary": summarise(rows, args.scales),
        "rows": rows,
    }
    output = (Path(args.output) if args.output else ROOT / "results/jenga/bench_eval" /
              f"{Path(args.model).stem}_action_branch.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["test"]["1.0"], indent=2))


if __name__ == "__main__":
    main()
