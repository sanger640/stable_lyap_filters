import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from local_persistent_branch import detect_local_persistent_branch, gap_bic_evidence


def test_smooth_curve_does_not_create_persistent_branch():
    strengths = np.linspace(-.5, .5, 31)
    smooth = strengths + .2 * strengths ** 2
    values = np.stack([smooth, 1.3 * smooth], axis=1)
    responses = np.stack([values, 1.2 * values], axis=1)
    result = detect_local_persistent_branch(strengths, responses)
    assert result.status == "not_resolved_on_grid"


def test_sustained_step_brackets_nearest_gap():
    strengths = np.linspace(-.5, .5, 31)
    branch = (strengths >= .15).astype(float)
    base = strengths + .1 * strengths ** 2
    values = np.stack([base + 2 * branch, .5 * base - branch], axis=1)
    responses = np.stack([values, 1.1 * values], axis=1)
    result = detect_local_persistent_branch(strengths, responses)
    assert result.status == "bracketed"
    assert result.direction == 1
    assert result.lower_abs_coefficient <= .15 <= result.upper_abs_coefficient


def test_one_tail_only_step_is_not_persistent():
    strengths = np.linspace(-.5, .5, 31)
    branch = (strengths >= .15).astype(float)
    first = np.stack([strengths + 2 * branch, strengths - branch], axis=1)
    second = np.stack([strengths, .5 * strengths], axis=1)
    responses = np.stack([first, second], axis=1)
    assert detect_local_persistent_branch(strengths, responses).status == "not_resolved_on_grid"


def test_gap_requires_three_points_per_side():
    strengths = np.linspace(-.5, .5, 9)
    values = strengths[:, None]
    assert gap_bic_evidence(strengths, values, 1) is None
    assert gap_bic_evidence(strengths, values, 8) is None
