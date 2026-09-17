import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from jenga_predictive_regime_probe import (HORIZON, STRENGTHS, physical_oracle,
                                           probe_chunk, score_state)  # noqa: E402


def test_probe_ramps_position_but_preserves_gripper():
    chunk = np.zeros((HORIZON, 4), np.float32)
    chunk[:, 3] = np.arange(HORIZON)
    displaced = probe_chunk(chunk, np.array([.008, 0, 0]), 1)
    assert np.isclose(displaced[0, 0], .001)
    assert np.isclose(displaced[-1, 0], .008)
    assert np.array_equal(displaced[:, 3], chunk[:, 3])


def test_oracle_requires_tilt_at_both_short_tails():
    tilts = np.zeros((3, len(STRENGTHS), 3), np.float32)
    tilts[0, -1, 1:] = [50, 30]
    assert physical_oracle(tilts)["mixed_directions"] == 0
    tilts[0, -1, 2] = 50
    assert physical_oracle(tilts)["mixed_directions"] == 1


def test_predictive_score_distinguishes_smooth_and_persistent_step():
    values = np.zeros((3, len(STRENGTHS), 3, 27), np.float32)
    values[:, :, :, 0] = STRENGTHS[None, :, None]
    axes = np.eye(3)
    assert not score_state(values, STRENGTHS, axes)["alarm"]
    values[0, :, :, 1] = (STRENGTHS[:, None] > .25) * 2
    score = score_state(values, STRENGTHS, axes)
    assert score["alarm"]
    assert score["margin"]["radial_interval"] == [.25, .375]
