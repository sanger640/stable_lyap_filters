import sys
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_short_held_tails import chunk_starts, model_action_window


def test_chunk_starts_use_frame_two_as_first_current_observation():
    assert chunk_starts(27) == [2, 10, 18]


def test_model_action_window_has_two_history_eight_chunk_and_tail():
    actions = np.arange(80, dtype=np.float32).reshape(20, 4)
    window = model_action_window(actions, start=2, tail=5)
    assert window.shape == (15, 4)
    np.testing.assert_array_equal(window[:10], actions[:10])
    np.testing.assert_array_equal(window[-5:], np.repeat(actions[9:10], 5, axis=0))


def test_model_action_window_rejects_incomplete_chunk():
    with pytest.raises(ValueError):
        model_action_window(np.zeros((9, 4)), start=2, tail=0)
