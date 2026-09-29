from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d11_fixed_expert_router import assignment, event_features, gate


def test_assignment_uses_complete_curve_error():
    target = torch.ones(2, 3, 4)
    curves = torch.zeros(2, 3, 3, 4)
    curves[0, 1] = target[0]; curves[1, 2] = target[1]
    assert assignment(curves, target).tolist() == [1, 2]


def test_event_features_expose_creation_and_loss():
    states = np.zeros((1, 4, 61), np.float32); actions = np.zeros((1, 3, 4), np.float32)
    states[0, 1:3, 45] = 1.
    norms = (np.zeros(45), np.ones(45), np.zeros(4), np.ones(4))
    features = event_features(states, actions, norms)
    assert features.shape == (1, 3, 142)
    assert features[0, 0, 114] == 1.  # first contact-creation channel
    assert features[0, 2, 126] == 1.  # first contact-loss channel


def test_router_gate_requires_gap_specificity_recall_and_error():
    def arm(mse, boundary_added, final_added, boundary_recall, final_recall):
        return {"mean_mse": mse, "nested": {
            "boundary": {"added_positive_rate": boundary_added, "recall": boundary_recall},
            "alarm": {"added_positive_rate": final_added, "recall": final_recall}}}
    config = {"minimum_fraction_of_d10_map_to_oracle_mse_gap_closed": .25,
              "boundary_added_positive_reduction_from_d9b": .1,
              "final_added_positive_reduction_from_d9b": .1,
              "maximum_boundary_recall_loss_from_d9b": .05,
              "maximum_final_recall_loss_from_d9b": .05}
    control = arm(.53, .6, .4, .9, .8)
    routed = arm(.48, .45, .25, .86, .76)
    assert gate(control, .62, .33, routed, config)["passes"]
    routed["mean_mse"] = .56
    assert not gate(control, .62, .33, routed, config)["passes"]
