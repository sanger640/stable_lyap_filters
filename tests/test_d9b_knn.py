from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_knn import episode_folds, knn_predict, interpretation


def test_episode_folds_are_disjoint_and_deterministic():
    episodes = np.asarray(["1", "1", "2", "3", "4", "5", "6"])
    folds = episode_folds(episodes, 3)
    assert [fold.tolist() for fold in folds] == [[0, 1, 4], [2, 5], [3, 6]]
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(episodes)))


def test_knn_uniform_and_inverse_distance():
    target = np.asarray([[0.], [2.], [10.]])
    distance = np.asarray([[1., 2., 9.]])
    np.testing.assert_allclose(knn_predict(distance, target, 2, "uniform"), [[1.]])
    np.testing.assert_allclose(knn_predict(distance, target, 2, "inverse_distance"), [[2 / 3]])


def test_knn_zero_distance_uses_only_exact_matches():
    target = np.asarray([[2.], [4.], [100.]])
    distance = np.asarray([[0., 0., 1.]])
    np.testing.assert_allclose(knn_predict(distance, target, 3, "inverse_distance"), [[3.]])


def test_interpretation_requires_curve_and_monitor_evidence():
    baseline = {"mean_mse": 1., "boundary_agreement": .8, "boundary_added": .2,
                "final_agreement": .7, "final_added": .1}
    good = {"mean_mse": .8, "boundary_agreement": .8, "boundary_added": .2,
            "final_agreement": .6, "final_added": .1}
    assert interpretation(good, baseline)["local_rule_beats_d9b"]
    good["boundary_added"] = .3
    assert not interpretation(good, baseline)["local_rule_beats_d9b"]
