from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_history_factorial import history_for_episodes, input_normalizers, pilot_gate


def test_history_selection_preserves_source_order_and_alignment():
    history = {"episode_id": np.asarray(["2", "1", "2"]),
               "state_history": np.arange(3 * 4 * 61).reshape(3, 4, 61),
               "action_history": np.arange(3 * 3 * 4).reshape(3, 3, 4)}
    states, actions = history_for_episodes(history, ["2"], history["state_history"][[0, 2], -1])
    assert np.array_equal(states, history["state_history"][[0, 2]])
    assert np.array_equal(actions, history["action_history"][[0, 2]])


def test_history_normalizers_are_finite_for_constant_channels():
    states = np.ones((3, 4, 61), np.float32)
    actions = np.ones((3, 3, 4), np.float32)
    values = input_normalizers(states, actions)
    assert all(np.all(np.isfinite(value)) for value in values)
    assert np.all(values[1] > 0) and np.all(values[3] > 0)


def test_factorial_gate_requires_specificity_recall_error_and_symmetry():
    def arm(mse, boundary_added, final_added, boundary_recall, final_recall):
        return {"mean_mse": mse, "nested": {
            "boundary": {"added_positive_rate": boundary_added, "recall": boundary_recall},
            "alarm": {"added_positive_rate": final_added, "recall": final_recall}}}
    config = {"boundary_added_positive_reduction": .1,
              "final_added_positive_reduction": .1,
              "maximum_boundary_recall_loss": .05,
              "maximum_final_recall_loss": .05,
              "endpoint_and_level_symmetry_max_abs": 1e-5}
    snapshot = arm(.6, .4, .3, .9, .8)
    temporal = arm(.5, .25, .15, .86, .76)
    assert pilot_gate(snapshot, temporal, 1e-6, config)["passes"]
    temporal["nested"]["alarm"]["added_positive_rate"] = .25
    assert not pilot_gate(snapshot, temporal, 1e-6, config)["passes"]
