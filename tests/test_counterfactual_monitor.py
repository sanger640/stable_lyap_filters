import numpy as np
import pytest

from counterfactual_monitor import persistent_mode_alarm


def separated(labels, dims=6):
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, (len(labels), dims))
    x[np.asarray(labels, bool), 0] += 3.0
    return x


def test_persistent_modes_alarm_without_calibration_data():
    labels = np.r_[np.zeros(32), np.ones(32)]
    result = persistent_mode_alarm(separated(labels), separated(labels))
    assert result.alarm
    assert result.early["bic_prefers_two"]
    assert result.late["minority"] == 32


def test_changed_partition_is_not_persistent():
    early = np.r_[np.zeros(32), np.ones(32)]
    late = early.copy(); late[[0, 32]] = late[[32, 0]]
    assert not persistent_mode_alarm(separated(early), separated(late)).alarm


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        persistent_mode_alarm(np.zeros((4, 2)), np.zeros((5, 2)))
