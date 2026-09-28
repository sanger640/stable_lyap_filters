import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_regime_monitor_v0_test_analysis",
    ROOT / "eval/jenga_regime_monitor_v0_test_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wilson_interval_contains_observed_rate():
    low, high = MODULE.wilson(55, 84)
    assert low < 55 / 84 < high


def test_analysis_counts_alarm_burden():
    rows = [
        {"class": "topple_fork", "episode_id": "1", "initial_alarm": True,
         "boundary_refined_alarm": True, "alarm": True},
        {"class": "quiet", "episode_id": "1", "initial_alarm": False,
         "boundary_refined_alarm": False, "alarm": False},
    ]
    # Stage attrition is intentionally tied to the immutable benchmark totals, but generic burden
    # and per-class summaries remain testable on a minimal fixture.
    result = MODULE.analyse({"rows": rows})
    assert result["intervention_burden"]["alarmed_states"] == 1
    assert result["intervention_burden"]["episodes_with_alarm"] == 1
