from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d6_memorization import capacity_gate, interpret, selected_rows


def evaluation(passing=True):
    good = 0.0 if passing else 0.1
    agreement = 1.0 if passing else 0.5
    return {
        "continuous": {"response": good, "topology": good, "commitment": good},
        "candidate": {"agreement": agreement, "mean_partition_agreement": agreement},
        "nested": {
            "boundary": {"agreement": agreement},
            "alarm": {"agreement": agreement},
        },
        "checks": {"permutation_max_abs": 0.0},
    }


def test_selected_rows_uses_first_row_for_each_episode():
    episodes = np.asarray(["8", "3", "8", "5", "3"])
    assert selected_rows(episodes, ["3", "8", "5"]).tolist() == [1, 0, 3]


def test_capacity_gate_requires_every_frozen_condition():
    spread = {"action": 1.0, "early_hold": 0.75, "late_hold": 1.25}
    assert capacity_gate(evaluation(), spread)["passes"]
    failed = capacity_gate(evaluation(False), spread)
    assert not failed["passes"]
    assert not failed["checks"]["candidate_decision"]


def test_interpretation_distinguishes_capacity_outcomes():
    passed = {"passes": True}; failed = {"passes": False}
    assert interpret({"independent_direct": passed, "set_conditioned": passed}) == (
        "both_representable_data_generalization")
    assert interpret({"independent_direct": passed, "set_conditioned": failed}) == (
        "independent_only_set_conditioning_collapse")
    assert interpret({"independent_direct": failed, "set_conditioned": passed}) == (
        "set_only_joint_context_required")
    assert interpret({"independent_direct": failed, "set_conditioned": failed}) == (
        "neither_passes_architecture_objective_optimization")
