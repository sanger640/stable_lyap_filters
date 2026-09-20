"""Does a W5 model reproduce the topples in its OWN training rollouts?

Generalisation cannot be read from a test score alone when the model has not fitted the training
data. This rolls each recorded training window forward from its true start state with the same
mean rollout the monitor uses, and counts new neighbour topples (tilt >= TOPPLE_DEG, excluding
neighbours already down at the chunk start) against the simulator's own.

Reported per model: recall on training rollouts that truly topple, the false rate on those that do
not, and the mean ending position error. Topple labels are used only to grade.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from state_dynamics import rollout  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import TOPPLE_DEG  # noqa: E402
from jenga_w5_eval import load_model, neighbour_tilt_deg  # noqa: E402
from jenga_w5_train import start_from_full_state  # noqa: E402


def episode_counts(path, model, scale, device):
    """-> (true, predicted) new-topple flags per rollout, and the ending position error in mm."""
    data = np.load(path, allow_pickle=False)
    traces, start_state, windows = data["traces"], data["start_state"], data["actions"]
    n_states, n_probes, steps = traces.shape[:3]
    start = start_from_full_state(start_state)
    states = np.repeat(start[:, None], n_probes, axis=1).reshape(-1, 61)
    actions = np.repeat(windows[:, :, 2:], model.substeps, axis=2).reshape(-1, steps, 4)
    with torch.no_grad():
        predicted = rollout(model, torch.as_tensor(states, device=device),
                            torch.as_tensor(actions, device=device), scale)[:, -1].cpu().numpy()
    truth = traces.reshape(-1, steps, 61)[:, -1]
    # A neighbour already down at the chunk start is not a topple this chunk caused.
    pose_only = np.concatenate(
        [start_state[..., 0:27], np.zeros(start_state.shape[:-1] + (34,), np.float32)], -1)
    eligible = np.repeat((neighbour_tilt_deg(pose_only) < TOPPLE_DEG)[:, None],
                         n_probes, axis=1).reshape(-1, 2)
    fell = lambda s: ((neighbour_tilt_deg(s) >= TOPPLE_DEG) & eligible).any(-1)  # noqa: E731
    return fell(truth), fell(predicted), 1000 * np.abs(predicted[:, :9] - truth[:, :9]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "results/jenga/trace_fine"))
    ap.add_argument("--files", type=int, default=60, help="training npz files to read")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(args.model, device)
    truth, predicted, errors = [], [], []
    for path in sorted(Path(args.data).glob("ep*_seed*.npz"))[:args.files]:
        true_flags, predicted_flags, error = episode_counts(path, model, scale, device)
        truth.append(true_flags); predicted.append(predicted_flags); errors.append(error)
    truth, predicted = np.concatenate(truth), np.concatenate(predicted)
    result = {"model": args.model, "rollouts": int(truth.size),
              "true_topples": int(truth.sum()),
              "recall_on_training_rollouts": float(predicted[truth].mean()),
              "false_rate": float(predicted[~truth].mean()),
              "ending_position_error_mm": float(np.mean(errors))}
    print(result)
    if args.output:
        Path(args.output).write_text(__import__("json").dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
