import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from basins import ProprioResidualizer, dissent_count, fit_basin_model, known_coverage
from jenga_runtime import coherent_action_probes


def test_hdbscan_finds_imbalanced_basins_and_rejects_far_noise():
    rng = np.random.default_rng(4)
    x = np.vstack([
        rng.normal((-4, 0), 0.12, (90, 2)),
        rng.normal((0, 3), 0.10, (25, 2)),
        rng.normal((4, 0), 0.11, (15, 2)),
    ]).astype(np.float32)
    model = fit_basin_model(x, pca_dim=2, min_cluster_fraction=0.05)
    assert model.n_clusters == 3
    assert model.coverage > 0.95
    assert model.predict(np.array([[20.0, 20.0]], np.float32))[0] == -1


def test_noise_is_excluded_from_dissent_vote():
    assert dissent_count([0, 0, 0, 1, -1, -1], n_clusters=2) == 1
    assert dissent_count([-1, -1], n_clusters=2) == 0
    assert known_coverage([0, 0, 0, 1, -1, -1], n_clusters=2) == 4 / 6


def test_proprio_residualizer_removes_linear_and_quadratic_arm_signal():
    rng = np.random.default_rng(8)
    p = rng.normal(size=(80, 4)).astype(np.float32)
    design = ProprioResidualizer.design(p)
    signal = design @ rng.normal(size=(design.shape[1], 12)).astype(np.float32)
    residualizer = ProprioResidualizer.fit(signal, p)
    assert np.abs(residualizer.transform(signal, p)).max() < 2e-4


def test_jenga_probes_scale_path_displacement_and_pin_gripper():
    actions = np.array([[0.4, 0.0, 0.5, 0.0],
                        [0.5, 0.1, 0.4, 1.0]], dtype=np.float32)
    probes = coherent_action_probes(actions, n=8, eps=0.1, seed=2)
    assert probes.shape == (8, 2, 4)
    np.testing.assert_allclose(probes[:, 0], np.repeat(actions[None, 0], 8, axis=0))
    np.testing.assert_allclose(probes[:, :, 3], np.repeat(actions[None, :, 3], 8, axis=0))
    # Every spatial coordinate is changed by the same scalar multiple of its path displacement.
    ratio_x = (probes[:, 1, 0] - actions[1, 0]) / (actions[1, 0] - actions[0, 0])
    ratio_y = (probes[:, 1, 1] - actions[1, 1]) / (actions[1, 1] - actions[0, 1])
    np.testing.assert_allclose(ratio_x, ratio_y, atol=1e-6)
