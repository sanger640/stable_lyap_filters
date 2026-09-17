import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_j2_j3_predicted_basins import (analyse, best_agreement, checkpoint_indices,
                                          label_stability, separation_ratio)


def test_checkpoint_indices_span_chunk_end_to_final_frame():
    idx = checkpoint_indices(101, horizon=8, fractions=(0.0, 0.5, 1.0))
    np.testing.assert_array_equal(idx, [10, 55, 100])


def test_agreement_is_permutation_invariant_and_ignores_noise():
    score, coverage = best_agreement([1, 1, 0, 0, -1], [0, 0, 1, 1, 0])
    assert score == 1.0
    assert coverage == 0.8


def test_label_stability_handles_independent_cluster_ids():
    score, coverage = label_stability([0, 0, 1, 1, -1], [1, 1, 0, 0, 0])
    assert score == 1.0
    assert coverage == 0.8


def test_separation_ratio_is_large_for_two_compact_groups():
    x = np.array([[0.0], [0.1], [5.0], [5.1]], np.float32)
    assert separation_ratio(x, [0, 0, 1, 1]) > 40


def test_analysis_reports_final_cluster_composition_and_error_ratio():
    endings = np.array([
        [[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]], [[0.3, 0.0]],
        [[5.0, 0.0]], [[5.1, 0.0]], [[5.2, 0.0]], [[5.3, 0.0]],
    ], np.float32)
    data = {
        "endings": endings,
        "proprio": np.zeros((8, 1, 1), np.float32),
        "episode_ids": np.array([str(i) for i in range(8)]),
        "tail_fractions": np.array([1.0], np.float32),
        "target_indices": np.tile([[3]], (8, 1)),
        "one_step_rmse": np.full(8, 0.1, np.float32),
        "one_step_nrmse": np.full(8, 0.2, np.float32),
    }
    labels = {str(i): {"outcome": "success" if i < 4 else "failure",
                       "peak_tilt_deg": float(i)} for i in range(8)}
    result = analyse(data, labels, [1], [0.25])
    final = result["configurations"]["pca1_min0.25"]["checkpoints"][-1]
    assert sum(x["intact"] + x["toppled"] for x in final["cluster_composition"]) \
        + sum(final["noise_composition"].values()) == 8
    assert final["one_step_to_basin_ratio"] > 0
