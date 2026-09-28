from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recoverability_monitor import reachable_overlap, recoverability_alarm


def sets(offset, corrections=7):
    values = np.linspace(-1., 1., corrections)
    pair = np.zeros((2, corrections, 45))
    pair[0, :, 0] = values
    pair[1, :, 0] = values + offset
    return pair


def test_overlapping_reachable_sets_are_recoverable():
    cross, resolution, ratio = reachable_overlap(sets(.1))
    assert cross < resolution
    assert ratio < 1.


def test_separated_reachable_sets_are_unrecoverable():
    cross, resolution, ratio = reachable_overlap(sets(4.))
    assert cross > resolution
    assert ratio > 1.


def test_identical_immobile_sets_are_recoverable():
    pair = np.zeros((2, 7, 45))
    assert reachable_overlap(pair)[2] == 0.


def test_majority_requires_boundary_and_nonoverlap_on_same_pairs():
    outcomes = np.stack([sets(4.), sets(3.), sets(.1)])
    result = recoverability_alarm(outcomes, np.ones(3), np.ones(3))
    assert result.alarm
    assert result.unrecoverable_pairs == 2


def test_without_local_boundary_no_alarm():
    outcomes = np.stack([sets(4.), sets(4.), sets(4.)])
    assert not recoverability_alarm(outcomes, -np.ones(3), -np.ones(3)).alarm
