import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from jenga_probe_comparison_videos import chosen_indices


def test_boundary_shows_distinct_adjacent_outcomes_and_nominal_probe():
    scalars = np.linspace(-1, 1, 6)
    outcomes = np.array([False, False, False, True, True, True])
    indices = chosen_indices("boundary", scalars, outcomes)
    assert indices[:2] == [2, 3]
    assert len(set(indices)) == 3
    assert outcomes[indices[0]] != outcomes[indices[1]]


def test_moving_safe_rejects_any_toppled_probe():
    with pytest.raises(ValueError, match="toppled"):
        chosen_indices("moving_safe", np.array([-1, 0, 1]),
                       np.array([False, True, False]))
