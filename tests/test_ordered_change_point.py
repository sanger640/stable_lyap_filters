import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from ordered_change_point import detect_ordered_jump, robust_component_scale


def smooth_response(s):
    return np.stack([s + 0.4 * s ** 2, np.sin(s), 0.2 * s ** 3], axis=1)


def test_smooth_nonlinearity_does_not_alarm():
    s = np.linspace(-2.3, 2.3, 50)
    result = detect_ordered_jump(s, smooth_response(s), degree=3)
    assert not result.alarm


def test_continuous_kink_does_not_alarm():
    s = np.linspace(-2.3, 2.3, 50)
    y = np.stack([s + 2 * np.maximum(s - 0.2, 0),
                  s ** 2 - np.maximum(s - 0.2, 0)], axis=1)
    result = detect_ordered_jump(s, y, degree=3)
    assert not result.alarm


def test_discontinuity_is_detected_and_localised():
    s = np.linspace(-2.3, 2.3, 50)
    y = smooth_response(s)
    y[s > 0.35] += np.array([2.0, -1.0, 0.5])
    result = detect_ordered_jump(s, y, degree=3)
    assert result.alarm
    assert abs(result.split_scalar - 0.35) < 0.12
    assert result.jump_evidence_bic > 0


def test_edge_jump_is_outside_supported_search():
    s = np.linspace(-2.3, 2.3, 50)
    y = smooth_response(s)
    y[s > 2.0] += 5.0
    result = detect_ordered_jump(s, y, degree=3, min_side=8)
    assert not result.alarm


def test_missing_probes_reduce_coverage():
    s = np.linspace(-2.3, 2.3, 50)
    y = smooth_response(s)
    y[:5] = np.nan
    result = detect_ordered_jump(s, y)
    assert result.valid_probes == 45
    assert result.coverage == 0.9


def test_component_scale_has_nonzero_floor():
    x = np.zeros((4, 50, 2))
    x[..., 0] = np.linspace(0, 1, 50)
    scale = robust_component_scale(x)
    assert np.all(scale > 0)
    assert scale[1] == 0.1 * scale[0]
