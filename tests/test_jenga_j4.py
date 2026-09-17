import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_j4_hedging import axis_cosine, nearest_centroid_accuracy, outcome_axis


def test_outcome_axis_reports_rms_distance_and_direction():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    rms, direction = outcome_axis(points, [0, 0, 1, 1])
    assert np.isclose(rms, 2.0)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_axis_cosine_detects_preserved_and_reversed_axes():
    assert np.isclose(axis_cosine([1, 0], [2, 0]), 1.0)
    assert np.isclose(axis_cosine([1, 0], [-2, 0]), -1.0)


def test_nearest_centroid_diagnostic_counts_predictions():
    train = np.array([[0.0], [0.2], [5.0], [5.2]])
    query = np.array([[0.1], [0.3], [4.9], [5.1]])
    accuracy, counts = nearest_centroid_accuracy(train, query, [0, 0, 1, 1])
    assert accuracy == 1.0
    assert counts == [2, 2]
