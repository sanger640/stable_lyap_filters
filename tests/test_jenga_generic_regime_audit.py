import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_generic_regime_audit", ROOT / "eval/jenga_generic_regime_audit.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reconstruct_endpoints_halves_the_selected_side():
    snippets = np.zeros((2, 8, 3), np.float32)
    snippets[1] = 1
    pairs = [{"probe_indices": [0, 1], "midpoint_sides": [0, 1]}]
    endpoints = MODULE.reconstruct_endpoints(snippets, pairs)[0]
    assert np.allclose(endpoints[0], .5)
    assert np.allclose(endpoints[1], .75)


def test_trace_metrics_detect_persistent_anonymous_contact_difference():
    left = np.zeros((38, 61), np.float32)
    right = np.zeros((38, 61), np.float32)
    right[10:20, 45] = 1
    metrics = MODULE.trace_pair_metrics(left, right)
    assert metrics["persistent_contact_difference"]
    assert metrics["contact_difference_max_run"] == 10
    assert metrics["first_contact_difference_step"] == 10


def test_trace_metrics_keep_pose_and_contact_channels_separate():
    left = np.zeros((38, 61), np.float32)
    right = np.zeros((38, 61), np.float32)
    right[:, 0] = .05
    metrics = MODULE.trace_pair_metrics(left, right)
    assert metrics["pose_gap_final"] == 1.0
    assert metrics["contact_difference_fraction"] == 0.0


def test_state_summary_uses_majority_counts_without_defining_an_alarm():
    pair = {"persistent_contact_difference": True, "late_contact_difference": False,
            "contact_difference_fraction": .1, "contact_difference_step_fraction": .2,
            "pose_gap_final": 1., "pose_gap_final_over_max": .5, "velocity_gap_max": 2.}
    summary = MODULE._state_summary([pair, pair, {**pair,
                                    "persistent_contact_difference": False}])
    assert summary["pairs_with_persistent_contact_difference"] == 2
    assert "alarm" not in summary
