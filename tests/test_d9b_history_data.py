from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_history_data import history_tail


def test_history_tail_returns_aligned_four_states_and_three_actions():
    states = np.arange(6 * 2).reshape(6, 2)
    actions = np.arange(5 * 3).reshape(5, 3)
    state_history, action_history = history_tail(states, actions, 4)
    np.testing.assert_array_equal(state_history, states[-4:])
    np.testing.assert_array_equal(action_history, actions[-3:])


def test_history_tail_rejects_misaligned_or_short_inputs():
    with pytest.raises(ValueError):
        history_tail(np.zeros((3, 2)), np.zeros((2, 3)), 4)
    with pytest.raises(ValueError):
        history_tail(np.zeros((4, 2)), np.zeros((2, 3)), 4)
