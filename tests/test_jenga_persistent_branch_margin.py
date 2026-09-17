import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from jenga_persistent_branch_margin import nearest_branch  # noqa: E402


def test_nearest_persistent_branch_brackets_original_nominal():
    strengths = np.array([-.3, -.2, -.1, 0, .1, .2, .3])
    labels = np.array([[2, 4], [2, 4], [0, 1], [0, 1], [0, 1], [3, 5], [3, 5]])
    result = nearest_branch(strengths, labels)
    assert result["status"] == "bracketed"
    assert result["direction"] == -1
    assert result["lower_abs_coefficient"] == .1
    assert result["upper_abs_coefficient"] == .2


def test_noise_abstains_and_cannot_become_alternative():
    strengths = np.array([-.2, -.1, 0, .1, .2])
    labels = np.array([[-1, -1], [-1, -1], [0, 1], [-1, -1], [-1, -1]])
    result = nearest_branch(strengths, labels)
    assert result["status"] == "not_found_within_grid"


def test_unassigned_nominal_abstains():
    strengths = np.array([-.1, 0, .1])
    labels = np.array([[0, 0], [-1, 0], [1, 1]])
    assert nearest_branch(strengths, labels)["status"] == "nominal_unassigned"


def test_transient_or_isolated_change_not_persistent():
    strengths = np.array([-.2, -.1, 0, .1, .2])
    labels = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]])
    assert nearest_branch(strengths, labels)["status"] == "edge_unconfirmed"


def test_transient_single_tail_change_has_no_branch():
    strengths = np.array([-.3, -.2, -.1, 0, .1, .2, .3])
    labels = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
    assert nearest_branch(strengths, labels)["status"] == "not_found_within_grid"


def test_edge_change_reported_as_unconfirmed():
    strengths = np.array([-.2, -.1, 0, .1, .2])
    labels = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]])
    result = nearest_branch(strengths, labels)
    assert result["status"] == "edge_unconfirmed"
    assert result["candidate_abs_coefficient"] == .2
