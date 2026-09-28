import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_action_boundary_refine", ROOT / "eval/jenga_action_boundary_refine.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_projector_uses_reference_only_and_maps_matching_point_identically():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(12, 2, 5))
    project = MODULE._projector(reference)
    assert np.allclose(project(reference[:1])[0], project(reference)[0])
    assert project(reference).shape == (12, 6)


def test_refinement_design_is_fixed_and_task_independent():
    assert MODULE.PAIRS == 3
    assert MODULE.REFINEMENTS == 5


def test_rescore_keeps_unrefined_states_quiet():
    rows = [{"alarm": True, "initial_alarm": False}]
    assert MODULE.rescore(rows)[0]["alarm"] is False
