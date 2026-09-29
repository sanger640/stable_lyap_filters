from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d18_commitment_robustness import (interpolate_missing, percentile_against_fit,
                                             robustness_curves, subset_summary)


def test_frozen_robustness_variants_have_expected_count_and_lengths():
    variants = robustness_curves(np.linspace(0, 1, 39))
    assert len(variants) == 9
    assert [len(variants[f"prefix_hold{hold}"]) for hold in (15, 20, 25)] == [24, 29, 34]
    assert all(len(values) == 39 for name, values in variants.items()
               if not name.startswith("prefix"))


def test_interpolation_exactly_recovers_a_linear_curve():
    curve = np.arange(39, dtype=float) * .25 + 2
    assert np.allclose(interpolate_missing(curve, [14, 15, 16]), curve)
    for values in robustness_curves(curve).values():
        if len(values) == 39:
            assert np.allclose(values, curve)


def test_margin_percentiles_and_sign_stability_summary():
    baseline = np.asarray([2., -1., 4.])
    variants = np.asarray([[1., 3.], [-2., -3.], [5., -1.]])
    percentiles = percentile_against_fit(np.abs(baseline), np.asarray([1., 2., 3., 4.]))
    assert np.allclose(percentiles, [.5, .25, 1.])
    result = subset_summary(np.asarray([True, True, True]), baseline, variants, percentiles)
    assert result["sign_stable_fraction"] == 2 / 3
    assert result["sign_unstable_fraction"] == 1 / 3
