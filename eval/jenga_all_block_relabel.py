"""Re-label the existing ordered-probe experiment with all three blocks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_failures import boundary_mask, failure_masks  # noqa: E402


def confusion(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp = int((truth & prediction).sum()); fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum()); tn = int((~truth & ~prediction).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-cache", default=str(
        ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--ordered-result", default=str(
        ROOT / "results/jenga/ordered_change_points_4x.json"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/all_block_relabel.json"))
    args = ap.parse_args()
    cache = dict(np.load(args.probe_cache, allow_pickle=False))
    ordered = json.loads(Path(args.ordered_result).read_text())
    alarms = np.asarray([row["alarm"] for row in ordered["actual_rows"]], bool)
    modes = failure_masks(cache["physical_peak_tilt"])
    summaries = {}
    for name, outcomes in modes.items():
        boundaries = boundary_mask(outcomes)
        summaries[name] = {
            "failed_probes": int(outcomes.sum()),
            "boundary_chunks": int(boundaries.sum()),
            "detector": confusion(boundaries, alarms),
            "boundary_rows": [
                {"episode_id": ordered["actual_rows"][i]["episode_id"],
                 "chunk_start": ordered["actual_rows"][i]["chunk_start"],
                 "failed_probes": int(outcomes[i].sum()),
                 "alarm": bool(alarms[i])}
                for i in np.flatnonzero(boundaries)],
        }
    result = {"failure_definition": {
        "neighbor": "left or right peak tilt >=45 deg",
        "middle": "middle peak tilt >=45 deg; falling below table added when min-z is available",
        "safe_middle_translation": "not a failure by itself"},
        "chunks": len(alarms), "probes_per_chunk": int(cache["physical_peak_tilt"].shape[1]),
        "summaries": summaries}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2)); print(f"wrote {output}")


if __name__ == "__main__":
    main()
