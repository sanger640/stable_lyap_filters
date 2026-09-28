import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_multiresolution_consequence",
    ROOT / "eval/jenga_multiresolution_consequence.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reconstruct_levels_follows_saved_bisection_sides():
    snippets = np.zeros((2, 8, 3), np.float32)
    snippets[1] = 1.
    details = [{"probe_indices": [0, 1], "midpoint_sides": [0, 1]}]
    levels = MODULE.reconstruct_levels(snippets, details)[0]
    assert np.allclose(levels[0, 0], 0.)
    assert np.allclose(levels[0, 1], 1.)
    assert np.allclose(levels[1, 0], .5)
    assert np.allclose(levels[1, 1], 1.)
    assert np.allclose(levels[2, 0], .5)
    assert np.allclose(levels[2, 1], .75)


def test_module_exposes_dev_evaluator():
    assert callable(MODULE.prepare)
    assert callable(MODULE.evaluate)
