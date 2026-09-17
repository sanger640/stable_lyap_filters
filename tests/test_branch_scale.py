import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from branch_scale import choose_larger_half, compare_scales  # noqa: E402


def test_smooth_separation_shrinks():
    result = compare_scales([[4, 2, 1], [8, 4, 2]])
    assert result.status == "shrinks_or_ambiguous"
    assert result.retained_fraction_by_tail == (.25, .25)


def test_persistent_jump_retains_separation():
    result = compare_scales([[4, 3.9, 3.8], [8, 7.8, 7.6]])
    assert result.status == "persistent_separation"


def test_both_tails_must_retain_separation():
    result = compare_scales([[4, 3.9, 3.8], [8, 4, 2]])
    assert result.status == "shrinks_or_ambiguous"


def test_refinement_follows_same_half_at_both_tails():
    left = np.array([[0.0], [0.0]])
    middle = np.array([[2.0], [3.0]])
    right = np.array([[2.1], [3.1]])
    assert choose_larger_half(left, middle, right) == 0
