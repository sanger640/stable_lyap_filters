from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d4_dev_analysis import gates  # noqa: E402


def test_dev_gate_requires_strict_recall_and_nondecreasing_agreement():
    matched = {"median_topple_alarms": 8, "median_quiet_alarms": 2,
               "median_physical_v0_agreement": .77}
    full = {"median_topple_alarms": 9, "median_quiet_alarms": 4,
            "median_physical_v0_agreement": .77}
    assert all(gates(matched, full).values())
    full["median_topple_alarms"] = 8
    assert not gates(matched, full)["recall"]
    full["median_topple_alarms"] = 9
    full["median_physical_v0_agreement"] = .76
    assert not gates(matched, full)["physical_agreement"]
