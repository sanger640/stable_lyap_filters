import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from intervention_policy import (ACTION_SCALES, choose_oracle_candidate,
                                 choose_robust_candidate, scaled_action_candidates)


def test_scaled_candidates_contract_xyz_and_hold_gripper():
    current = np.array([1., 2., 3., -1.], np.float32)
    chunk = np.array([[2., 4., 6., 1.], [3., 6., 9., 1.]], np.float32)
    candidates = scaled_action_candidates(chunk, current)
    assert np.allclose(candidates[0], chunk)
    assert np.allclose(candidates[2, :, :3], current[:3] + .5 * (chunk[:, :3] - current[:3]))
    assert np.allclose(candidates[-1, :, :3], current[:3])
    assert np.all(candidates[-1, :, 3] == current[3])


def test_robust_choice_minimises_pairs_then_deviation():
    evidence = [{"consequential_pairs": value} for value in (3, 1, 1, 0, 0)]
    selected = choose_robust_candidate(evidence)
    assert selected.scale == ACTION_SCALES[3]


def test_oracle_avoids_topple_then_keeps_nominal_action():
    grades = [(True, 50.), (False, 40.), (False, 20.), (False, 10.), (False, 5.)]
    assert choose_oracle_candidate(grades) == 1
