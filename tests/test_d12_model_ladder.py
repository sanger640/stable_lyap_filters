from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d12_model_ladder import arm_gate


def metrics(mse=.5, boundary_added=.4, boundary_recall=.8, final_added=.3, final_recall=.7):
    return {"mean_mse": mse, "nested": {
        "boundary": {"added_positive_rate": boundary_added, "recall": boundary_recall},
        "alarm": {"added_positive_rate": final_added, "recall": final_recall}}}


def test_gate_requires_both_specificity_gains_without_recall_loss():
    config = {"boundary_added_positive_reduction": .1, "final_added_positive_reduction": .1,
              "maximum_boundary_recall_loss": .05, "maximum_final_recall_loss": .05,
              "router_minimum_fraction_of_map_to_oracle_mse_gap_closed": .25}
    assert arm_gate(metrics(), metrics(.4, .3, .76, .2, .66), config)["passes"]
    assert not arm_gate(metrics(), metrics(.4, .35, .8, .2, .7), config)["passes"]
    routed = arm_gate(metrics(), metrics(.35, .3, .8, .2, .7), config,
                      oracle_mse=.2, map_mse=.5)
    assert routed["passes"]
    assert routed["comparisons"]["fraction_of_map_to_oracle_gap_closed"] == pytest.approx(.5)
