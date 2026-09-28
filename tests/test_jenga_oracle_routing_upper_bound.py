import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_oracle_routing_upper_bound.py"
SPEC = importlib.util.spec_from_file_location("jenga_oracle_routing_upper_bound", SCRIPT)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def test_balanced_assignment_uses_nearly_equal_capacities():
    values = np.arange(11, dtype=float)[:, None]
    centroids = np.asarray([[0.0], [5.0], [10.0]])

    assignment = oracle.balanced_assign(values, centroids)
    counts = np.bincount(assignment, minlength=3)

    assert counts.sum() == len(values)
    assert counts.max() - counts.min() <= 1


def test_balanced_kmeans_is_deterministic_and_uses_every_mode():
    values = np.concatenate([
        np.full((6, 2), -4.0),
        np.full((6, 2), 0.0),
        np.full((6, 2), 5.0),
    ])

    first_centroids, first_assignment, _ = oracle.balanced_kmeans(values, modes=3)
    second_centroids, second_assignment, _ = oracle.balanced_kmeans(values, modes=3)

    np.testing.assert_allclose(first_centroids, second_centroids)
    np.testing.assert_array_equal(first_assignment, second_assignment)
    np.testing.assert_array_equal(np.bincount(first_assignment, minlength=3), [6, 6, 6])


def test_future_informed_prototypes_reduce_synthetic_scale_error():
    truth = np.zeros((9, 5, 3), dtype=float)
    predicted = truth.copy()
    corrections = np.asarray([-1.0, 0.5, 2.0])
    for group, correction in enumerate(np.repeat(corrections, 3)):
        truth[group] += correction

    residual = (truth - predicted).reshape(9, -1)
    centroids, _, _ = oracle.balanced_kmeans(residual, modes=3)
    result = oracle.evaluate_split(predicted, truth, centroids)

    assert result["loss_reduction"]["global_scale_huber"] > 0.999
    assert set(result["assignment"]) == {0, 1, 2}
