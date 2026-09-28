import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_regime_monitor_v0_test", ROOT / "eval/jenga_regime_monitor_v0_test.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_summary_keeps_detector_stages_separate():
    rows = [
        {"class": "topple_fork", "initial_alarm": True,
         "boundary_refined_alarm": True, "alarm": True},
        {"class": "topple_fork", "initial_alarm": True,
         "boundary_refined_alarm": False, "alarm": False},
        {"class": "quiet", "initial_alarm": False,
         "boundary_refined_alarm": False, "alarm": False},
    ]
    result = MODULE.candidate_summary(rows)
    assert result["topple_fork"] == {
        "states": 2, "initial_candidates": 2, "boundary_candidates": 1, "v0_alarms": 1}
    assert result["quiet"]["states"] == 1


def test_test_evaluator_has_no_force_or_limit_escape_hatch():
    source = (ROOT / "eval/jenga_regime_monitor_v0_test.py").read_text()
    assert "--force" not in source
    assert "--limit" not in source
