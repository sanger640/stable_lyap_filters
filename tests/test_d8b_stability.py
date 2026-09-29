from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d8b_stability import aggregate


def row(passes, early):
    errors = {name: 0. for name in (
        "early_gap", "full_gap", "curve_action", "curve_early_hold", "curve_late_hold")}
    errors["early_gap"] = early
    return {"gate": {"passes": passes}, "curve_errors": errors,
            "nested": {"boundary": {"agreement": 1.}, "alarm": {"agreement": 1.}}}


def test_replication_requires_every_seed():
    result = aggregate([row(True, 1.), row(True, 2.), row(False, 3.)])
    assert not result["all_seeds_pass"]
    assert result["passing_seeds"] == 2
    assert result["curve_error_ranges"]["early_gap"] == {"min": 1., "median": 2., "max": 3.}
