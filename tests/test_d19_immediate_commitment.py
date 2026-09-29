from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d19_immediate_commitment import (IMMEDIATE_LENGTH, formulation_summary,
                                             late_sampling_invariant)


def test_immediate_window_is_zero_action_and_exactly_five_held_samples():
    assert IMMEDIATE_LENGTH == 14


def test_immediate_commitment_input_is_invariant_to_all_frozen_late_variants():
    curve = np.sin(np.linspace(0, 3, 39)) + np.linspace(0, 1, 39)
    assert late_sampling_invariant(curve)


def test_formulation_summary_reports_retained_and_added_signs():
    summary = formulation_summary(
        full=np.asarray([2., 1., -1., -2.]),
        immediate=np.asarray([3., -1., 2., -3.]),
        persistence=np.asarray([1., 1., -1., -1.]),
    )
    assert summary["full_positive_retention"] == .5
    assert summary["added_positive_rate_among_full_negatives"] == .5
    assert summary["full_and_persistent"] == 2
    assert summary["immediate_and_persistent"] == 1
