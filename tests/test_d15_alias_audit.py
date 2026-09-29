from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d14_object_graph import permute_objects
from jenga_d15_alias_audit import (diagnostic_marker, neighbor_metrics, pairwise_rms,
                                   permutation_invariant_scene_distance, graph_vectors)
from set_response_model import ObjectGraphJointMonitorEvidenceModel


def network():
    return ObjectGraphJointMonitorEvidenceModel(
        hidden=16, heads=4, layers=1, graph_rounds=1,
        action_horizon=8, curve_steps=39).eval()


def states():
    generator = torch.Generator().manual_seed(15)
    values = torch.randn(4, 61, generator=generator) * .02
    values[:, 45:57] = (torch.rand(4, 12, generator=generator) > .5).float()
    return values


def test_pairwise_rms_and_neighbor_sign_metrics():
    fit = np.asarray([[0.], [1.], [2.], [3.]])
    held = np.asarray([[.1], [2.9]])
    distance = pairwise_rms(held, fit)
    metrics = neighbor_metrics(distance, np.asarray([False, False, True, True]),
                               np.asarray([False, True]), 3)
    assert metrics["nearest_index"].tolist() == [0, 3]
    assert np.allclose(metrics["same_sign_purity"], 2 / 3)
    assert metrics["five_neighbor_correct"].tolist() == [True, True]


def test_raw_scene_distance_minimizes_object_relabelling():
    model = network(); fit = states()
    fit_vectors = graph_vectors(model, fit.numpy(), "cpu")
    held = permute_objects(fit, (2, 0, 1)).numpy()
    distance = permutation_invariant_scene_distance(model, held, fit_vectors, "cpu")
    assert np.all(np.min(distance, axis=1) < 1e-6)


def test_predeclared_markers_are_exclusive():
    base = {"count": 10, "raw_same_sign_purity_median": .4,
            "raw_opposite_to_same_ratio_median": 1.1,
            "learned_same_sign_purity_median": .9}
    assert diagnostic_marker(base) == "raw_snapshot_alias_pressure"
    base.update(raw_same_sign_purity_median=.9, learned_same_sign_purity_median=.4)
    assert diagnostic_marker(base) == "learned_representation_collapse"
    base.update(learned_same_sign_purity_median=.8)
    assert diagnostic_marker(base) == "mapping_or_coverage_failure"
