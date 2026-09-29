from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from program_b_timestep_audit import audit_episode, build_frame, summarise  # noqa: E402


def test_build_frame_is_last_graph_before_the_onset():
    assert build_frame(1, 10) == 0 and build_frame(10, 10) == 0 and build_frame(11, 10) == 1
    assert build_frame(50, 50) == 0 and build_frame(51, 50) == 5 and build_frame(99, 50) == 5


def test_onset_is_represented_only_if_near_at_the_build_frame():
    onsets = np.array([[37, 0, 1]])                               # built at 30 (20 ms) or 0 (100 ms)
    audit = np.array([[3, 0, 1, 0.004]])                          # frame 3 = substep 30, within 5 mm
    row = audit_episode(onsets, audit)[0]
    assert row["20ms"] and not row["100ms"]
    far = audit_episode(onsets, np.array([[3, 0, 1, 0.008]]))[0]
    assert not far["20ms"] and not far["100ms"]
    summary = summarise([row])
    assert summary["missed_20ms"] == 0.0 and summary["missed_100ms"] == 1.0
