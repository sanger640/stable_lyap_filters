from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9c_interaction_event import event_normalizers, pilot_gate


def test_event_normalizers_keep_constant_channels_finite():
    states = np.ones((4, 4, 61), np.float32)
    actions = np.ones((4, 3, 4), np.float32)
    values = event_normalizers(states, actions)
    assert all(np.all(np.isfinite(value)) for value in values)
    assert np.all(values[1] > 0) and np.all(values[3] > 0)


def test_event_gate_requires_specificity_recall_error_and_history_control():
    def arm(mse, boundary_added, final_added, boundary_recall, final_recall):
        return {"mean_mse": mse, "nested": {
            "boundary": {"added_positive_rate": boundary_added, "recall": boundary_recall},
            "alarm": {"added_positive_rate": final_added, "recall": final_recall}}}
    config = {"boundary_added_positive_reduction_from_snapshot": .1,
              "final_added_positive_reduction_from_snapshot": .1,
              "maximum_boundary_recall_loss_from_snapshot": .05,
              "maximum_final_recall_loss_from_snapshot": .05,
              "endpoint_and_level_symmetry_max_abs": 1e-5}
    snapshot = arm(.6, .5, .4, .9, .8)
    history = arm(.59, .45, .35, .87, .7)
    event = arm(.55, .35, .25, .86, .76)
    assert pilot_gate(snapshot, history, event, 1e-6, config)["passes"]
    event["nested"]["alarm"]["added_positive_rate"] = .36
    assert not pilot_gate(snapshot, history, event, 1e-6, config)["passes"]
