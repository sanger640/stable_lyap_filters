import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_pair_selection_crossover.py"
SPEC = importlib.util.spec_from_file_location("jenga_pair_selection_crossover", SCRIPT)
crossover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crossover)


def test_compact_existing_summary_keeps_counts_only():
    result = {"summary": {"dev": {"1.0": {
        "topple_fork": {"states": 23, "alarms": 18, "rate": 18 / 23},
        "quiet": {"states": 89, "alarms": 7, "rate": 7 / 89},
    }}}}

    compact = crossover.compact_existing_summary(result)

    assert compact == {
        "topple_fork": {"states": 23, "alarms": 18},
        "quiet": {"states": 89, "alarms": 7},
    }


def test_empty_class_summary_has_zero_counts():
    summary = crossover.summarize([])

    assert summary["topple_fork"] == {
        "states": 0, "initial_candidates": 0, "boundary_candidates": 0,
        "commitment_majority": 0, "persistence_majority": 0, "alarms": 0,
    }
