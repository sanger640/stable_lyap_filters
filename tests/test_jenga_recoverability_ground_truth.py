import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_recoverability_ground_truth",
    ROOT / "eval/jenga_recoverability_ground_truth.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_correction_library_is_symmetric_and_has_seven_members():
    rng = np.random.default_rng(3)
    offsets = MODULE.correction_offsets(rng.normal(size=(64, 8, 3)))
    assert offsets.shape == (7, 3)
    assert np.allclose(offsets[0], 0.)
    assert np.allclose(offsets[1], -offsets[2])
    assert np.allclose(offsets[3], -offsets[4])
    assert np.allclose(offsets[5], -offsets[6])


def test_module_exposes_dev_evaluator():
    assert callable(MODULE.prepare)
    assert callable(MODULE.evaluate)
