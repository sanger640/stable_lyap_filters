from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

import panda_push_fresh_test as fresh
import panda_push_dev_panel as dev


def test_fresh_candidate_specs_are_deterministic_and_do_not_repeat_dev_specs():
    first = fresh.candidate_specs(); second = fresh.candidate_specs()
    assert first == second and len(first) == sum(fresh.PROTOCOL["candidate_families"].values())
    old = {(row["family"], row["z"], row["end_x"], row["contact_y"],
            row["block_y"], row["veer_y"]) for row in dev.candidate_specs()}
    assert not any((row["family"], row["z"], row["end_x"], row["contact_y"],
                    row["block_y"], row["veer_y"]) in old for row in first)


def test_test_protocol_excludes_monitor_scores_from_selection():
    assert fresh.PROTOCOL["task_labels_in_monitor"] is False
    assert "monitor scores absent" in fresh.PROTOCOL["status"]
    assert fresh.PROTOCOL["workers"] == 1
