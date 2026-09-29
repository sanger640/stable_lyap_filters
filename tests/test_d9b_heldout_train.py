from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_heldout_train import median_summary, success_gate, zero_metrics


def test_zero_metrics_use_literal_zero_without_threshold():
    result = zero_metrics([0., 0., 1., 2.], [0., 1., 0., 2.])
    assert result == {"groups": 4, "reference_zero": 2, "predicted_zero": 2,
                      "tp": 1, "fn": 1, "fp": 1, "tn": 1,
                      "recall": .5, "false_zero_rate": .5}


def test_median_summary_and_gate_compare_with_frozen_d2():
    rows = []
    for boundary_recall, final_recall in ((.4, .2), (.5, .3), (.6, .4)):
        stage = lambda recall: {"agreement": .7, "recall": recall,
                                "added_positive_rate": .1}
        rows.append({"nested": {"boundary": stage(boundary_recall),
                                "commitment": stage(.5), "persistence": stage(.5),
                                "alarm": stage(final_recall)},
            "zero": {"recall": .8, "false_zero_rate": .1},
            "invariance": {"endpoint_swap_max_abs": 0., "level_permutation_max_abs": 1e-6}})
    summary = median_summary(rows)
    assert summary["boundary"]["recall"] == .5
    reference = {"boundary": {"recall": .25, "added_positive_rate": .14},
                 "alarm": {"recall": .05, "added_positive_rate": .08}}
    assert success_gate(summary, rows, reference)["passes"]


def test_gate_rejects_excess_added_positive_rate():
    stage = lambda recall, added: {"agreement": .7, "recall": recall,
                                   "added_positive_rate": added}
    rows = [{"nested": {"boundary": stage(.5, .3), "commitment": stage(.5, .1),
                         "persistence": stage(.5, .1), "alarm": stage(.3, .2)},
             "zero": {"recall": .8, "false_zero_rate": .1},
             "invariance": {"endpoint_swap_max_abs": 0., "level_permutation_max_abs": 0.}}
            for _ in range(3)]
    summary = median_summary(rows)
    reference = {"boundary": {"recall": .25, "added_positive_rate": .14},
                 "alarm": {"recall": .05, "added_positive_rate": .08}}
    assert not success_gate(summary, rows, reference)["passes"]
