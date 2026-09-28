import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_selective_router_dev.py"
SPEC = importlib.util.spec_from_file_location("jenga_selective_router_dev", SCRIPT)
dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev)


def test_pair_feature_uses_nominal_and_twice_endpoint_difference():
    start = np.arange(61, dtype=np.float32)
    snippets = np.zeros((3, 8, 3), dtype=np.float32)
    snippets[0] = .2
    windows = np.zeros((3, 40, 4), dtype=np.float32)
    windows[0, 2:10, :3] = 1.0 - snippets[0]
    windows[0, 2:10, 3] = .7
    windows[1, 2:10, :3] = .4
    windows[2, 2:10, :3] = .6

    feature = dev.pair_feature(start, windows, snippets, (1, 2))

    np.testing.assert_allclose(feature[:61], start)
    nominal = feature[61:93].reshape(8, 4)
    np.testing.assert_allclose(nominal[:, :3], 1.0)
    np.testing.assert_allclose(nominal[:, 3], .7)
    np.testing.assert_allclose(feature[93:].reshape(8, 3), .4)


def test_corrected_gaps_preserves_level_zero_and_applies_selected_phase():
    gaps = np.arange(1, 7, dtype=float)
    correction = np.zeros((5, 3), dtype=float)
    correction[:, 2] = np.log(2.0)

    corrected = dev.corrected_gaps(gaps, correction, phase=2)

    assert corrected[0] == gaps[0]
    np.testing.assert_allclose(corrected[1:], 2 * gaps[1:])


def test_nested_noises_follows_frozen_midpoint_sides():
    snippets = np.zeros((2, 8, 3), dtype=np.float32)
    snippets[1] = 1.0
    pair = {"probe_indices": [0, 1], "midpoint_sides": [0, 1]}

    levels = dev.nested_noises(snippets, pair)

    np.testing.assert_allclose(levels[:, :, 0, 0], [[0, 1], [.5, 1], [.5, .75]])
