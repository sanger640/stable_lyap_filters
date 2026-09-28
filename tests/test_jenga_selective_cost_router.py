import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_selective_cost_router.py"
SPEC = importlib.util.spec_from_file_location("jenga_selective_cost_router", SCRIPT)
selective = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selective)


def test_option_cost_prefers_identity_when_prediction_is_exact():
    truth = np.zeros((2, 5, 3), dtype=float)
    predicted = truth.copy()
    corrections = np.stack([np.zeros((5, 3)), np.ones((5, 3))])

    global_cost, local_cost = selective.option_cost_components(
        predicted, truth, corrections)

    np.testing.assert_allclose(global_cost[:, 0], 0.0)
    np.testing.assert_allclose(local_cost[:, 0], 0.0)
    assert np.all(global_cost[:, 1] > global_cost[:, 0])


def test_normalized_cost_equal_weights_global_and_local_components():
    global_cost = np.asarray([[2.0, 1.0]])
    local_cost = np.asarray([[3.0, 6.0]])

    cost = selective.normalized_option_cost(
        global_cost, local_cost, {"global": 2.0, "local": 3.0})

    np.testing.assert_allclose(cost, [[2.0, 2.5]])


def test_selective_router_outputs_one_cost_per_option():
    model = selective.SelectiveCostRouter(117, options=4)
    assert model(torch.zeros(7, 117)).shape == (7, 4)


def test_smooth_weak_selection_rate_counts_identity_and_weak_mode():
    truth = np.zeros((4, 5, 3), dtype=float)
    truth[:, -1] = np.log(.05)
    assignment = np.asarray([0, 1, 2, 3])

    assert selective.smooth_weak_selection_rate(assignment, truth) == .5
