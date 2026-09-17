import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from jenga_stage0_noise_oracle import detection_probability, grade, sample_snippets  # noqa: E402


def test_grade_requires_two_on_each_side():
    assert grade(np.zeros(64, bool)) == "unanimous_safe"
    assert grade(np.ones(64, bool)) == "unanimous_topple"
    one = np.zeros(64, bool); one[0] = True
    assert grade(one) == "weak_topple_minority"
    one[1] = True
    assert grade(one) == "mixed"
    assert grade(~np.eye(64, dtype=bool)[0]) == "weak_safe_minority"


def test_detection_probability_matches_plan():
    assert abs(detection_probability(.05) - .836) < .005
    assert detection_probability(0.0) == 0.0


def test_snippets_are_contiguous_windows():
    pools = [np.arange(30, dtype=float).reshape(10, 3), np.zeros((3, 3))]
    snippets = sample_snippets(pools, count=20, horizon=8, seed=1)
    assert snippets.shape == (20, 8, 3)
    assert np.all(np.diff(snippets[:, :, 0], axis=1) == 3)
