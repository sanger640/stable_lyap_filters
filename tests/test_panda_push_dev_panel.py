import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]


def load():
    spec = importlib.util.spec_from_file_location(
        "panda_push_dev_panel", ROOT / "eval" / "panda_push_dev_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_panda_push_dev_panel_and_result():
    module = load()
    assert module.verify()["protocol_sha256"] == (
        "20dc10ea8eb455fceb958e0c409f94fdeafacddfb577ef479db6b443742d1aed")
    assert module.RESULT.is_file()


def test_panda_push_dev_summary_has_expected_stage_diagnostic():
    import json
    module = load(); result = json.loads(module.RESULT.read_text())["summary"]
    assert result["topple_fork"]["final_alarms"] == 19
    assert result["safe_centered"]["final_alarms"] == 0
    assert result["contact_loss_control"]["final_alarms"] == 5
    assert result["overshoot_control"]["final_alarms"] == 0
