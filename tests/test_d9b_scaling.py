from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_scaling import nested_episode_subsets, pilot_gate


def test_nested_episode_subsets_are_deterministic_and_nested():
    episodes = [str(value) for value in range(10, 22)]
    subsets = nested_episode_subsets(episodes, 2, (.25, .5, .75, 1.))
    assert [len(subsets[value]) for value in (.25, .5, .75, 1.)] == [3, 6, 9, 12]
    assert set(subsets[.25]) < set(subsets[.5]) < set(subsets[.75]) < set(subsets[1.])
    assert subsets == nested_episode_subsets(episodes, 2, (.25, .5, .75, 1.))


def test_pilot_gate_passes_clear_monotone_scaling():
    aggregate = {
        "0.25": {"mean_mse": 1., "nested": {"boundary": {"added_positive_rate": .3},
                                               "alarm": {"added_positive_rate": .2}}},
        "0.5": {"mean_mse": .85, "nested": {"boundary": {"added_positive_rate": .25},
                                               "alarm": {"added_positive_rate": .18}}},
        "0.75": {"mean_mse": .75, "nested": {"boundary": {"added_positive_rate": .2},
                                                "alarm": {"added_positive_rate": .15}}},
        "1.0": {"mean_mse": .7, "nested": {"boundary": {"added_positive_rate": .2},
                                              "alarm": {"added_positive_rate": .1}}}}
    folds = [{"fraction": fraction, "fold": fold,
              "mean_mse": 1. - .2 * float(fraction)}
             for fraction in (.25, 1.) for fold in range(5)]
    assert pilot_gate(aggregate, folds)["passes"]


def test_pilot_gate_rejects_flat_error():
    aggregate = {str(value): {"mean_mse": 1., "nested": {
        "boundary": {"added_positive_rate": .2}, "alarm": {"added_positive_rate": .1}}}
        for value in (.25, .5, .75, 1.)}
    folds = [{"fraction": fraction, "fold": fold, "mean_mse": 1.}
             for fraction in (.25, 1.) for fold in range(5)]
    assert not pilot_gate(aggregate, folds)["passes"]
