import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from local_expansion import detect_local_expansion


def test_smooth_curve_does_not_alarm():
    s = np.linspace(-2, 2, 50)
    y = np.stack([s, s ** 2, np.sin(s), np.cos(s)], axis=1)
    result = detect_local_expansion(s, y, half_window=8)
    assert not result.alarm


def test_constant_response_has_zero_expansion():
    s = np.linspace(-2, 2, 50)
    y = np.repeat(np.array([[1.0, 2.0, 3.0]]), 50, axis=0)
    result = detect_local_expansion(s, y, half_window=8)
    assert not result.alarm
    assert result.peak_indices == ()


def test_two_steps_produce_two_local_peaks():
    s = np.linspace(-2, 2, 50)
    y = np.stack([s, .2 * s, -.3 * s, s ** 2], axis=1)
    y[s > -.7, 0] += 4
    y[s > .8, 1] -= 5
    result = detect_local_expansion(s, y, half_window=8)
    assert result.alarm
    assert len(result.peak_scalars) == 2
    assert abs(result.peak_scalars[0] + .7) < .15
    assert abs(result.peak_scalars[1] - .8) < .15


def test_scale_shape_is_checked():
    with np.testing.assert_raises(ValueError):
        detect_local_expansion([0, 1, 2, 3, 4, 5], np.ones((6, 2)),
                               half_window=3, scale=np.ones(3))
