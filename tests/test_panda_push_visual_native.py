from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"eval"))

import panda_push_visual_native as visual


def test_protocol_removes_physical_channel_semantics_without_calibration():
    assert visual.PROTOCOL["physical_channel_groups_used"] is False
    assert visual.PROTOCOL["reference_or_outcome_calibration"] is False
    assert visual.PROTOCOL["task_labels_in_monitor"] is False
