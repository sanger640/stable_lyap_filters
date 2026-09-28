"""Rescore rendered Panda futures with the proper visual-native v0 consequence stage.

Branch discovery, action-boundary refinement and all exact trajectories are reused unchanged from
the completed representation experiments.  Only the invalid physical-channel slicing of PCA axes
is removed.  The complete visual feature vector now forms one internally normalized metric space;
the frozen v0 BIC tests, zero decision boundary and majority rule remain identical.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"eval"))

from consequence_monitor import result_dict as consequence_result_dict
import panda_push_visual_v0 as baseline
from visual_consequence_monitor import visual_whole_trajectory_consequence_alarm

OUT = ROOT/"results/panda_block_push/visual_native"
RESULT = OUT/"visual_native_diagnostic.json"
ADAPTERS = {
    "full_frame_pca45": ROOT/"results/panda_block_push/visual_pca45/evaluated_states",
    "motion_pca45": ROOT/"results/panda_block_push/visual_regions/motion_pca45/evaluated_states",
    "block_pca45_oracle": ROOT/"results/panda_block_push/visual_regions/block_pca45/evaluated_states",
}
PHYSICAL = ROOT/"results/panda_block_push/dev_panel/v0_stage_diagnostic.json"

PROTOCOL = {
    "id": "panda-upright-push-visual-native-v0-dev",
    "status": "DEV-only visual-port diagnostic; not a holdout claim",
    "source_panel_sha256": baseline.PROTOCOL["source_panel_sha256"],
    "upstream": "reuse each representation's own frozen candidate partition, 3 refined pairs and 5 bisections",
    "visual_curve": "one RMS-normalized Euclidean separation curve over the complete PCA45 trajectory",
    "unchanged_logic": "boundary BIC, action exposure, commitment BIC, persistence BIC, zero boundary, pair majority",
    "physical_channel_groups_used": False,
    "reference_or_outcome_calibration": False,
    "task_labels_in_monitor": False,
}


def rescore_state(path, physical_row):
    cache = np.load(path, allow_pickle=False)
    previous = json.loads(str(cache["result_json"]))
    row = dict(previous)
    row["previous_physical_channel_adapter_alarm"] = bool(previous["alarm"])
    if not previous["initial_alarm"]:
        row["alarm"] = False
        return row
    boundary = previous["boundary_evidence"]
    evidence = visual_whole_trajectory_consequence_alarm(
        cache["dense_features"], cache["endpoints"][:, 0]-cache["endpoints"][:, 1],
        boundary["early_delta_bic"], boundary["full_delta_bic"])
    row.update(consequence_result_dict(evidence))
    row["physical_alarm"] = bool(physical_row["alarm"])
    return row


def evaluate():
    physical = json.loads(PHYSICAL.read_text())["rows"]
    reports = {}
    for name, directory in ADAPTERS.items():
        paths = [directory/f"state_{index:03d}.npz" for index in range(len(physical))]
        missing = [str(path) for path in paths if not path.exists()]
        if missing: raise FileNotFoundError(f"{name} is missing {len(missing)} state caches")
        rows = [rescore_state(path, physical_row) for path, physical_row in zip(paths, physical)]
        reports[name] = {"summary": baseline.summarise(rows, physical), "rows": rows}
    value = {"protocol": PROTOCOL,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "representations": reports}
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps({name: report["summary"] for name, report in reports.items()}, indent=2))
    return value


if __name__ == "__main__": evaluate()
