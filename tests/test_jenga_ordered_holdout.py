import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_ordered_holdout import binomial_interval, read_cache, write_cache


def test_holdout_cache_round_trip(tmp_path):
    rows = [{"episode_id": "4", "chunk_start": 2}]
    scalars = np.array([-1, 1], np.float32)
    actual = np.zeros((1, 2, 3), np.float32)
    physical = np.array([[False, True]])
    path = tmp_path / "holdout.npz"
    write_cache(path, rows, scalars, actual, physical)
    rows2, scalars2, actual2, physical2 = read_cache(path)
    assert rows2 == rows
    np.testing.assert_array_equal(scalars2, scalars)
    np.testing.assert_array_equal(actual2, actual)
    np.testing.assert_array_equal(physical2, physical)


def test_exact_binomial_interval_contains_rate():
    lower, upper = binomial_interval(3, 50)
    assert lower < 0.06 < upper
