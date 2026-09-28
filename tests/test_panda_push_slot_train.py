from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

from models.slot_dynamics import SlotDynamicsRepresentation
import panda_push_slot_train as training


def test_label_free_objective_is_finite_and_backpropagates():
    torch.manual_seed(4)
    model = SlotDynamicsRepresentation(input_dim=8, slot_dim=12, slots=3, iterations=1)
    descriptors = torch.randn(2, 2, 4, 16, 8)
    actions = torch.randn(2, 2, 3, 4)
    loss, terms = training.objective(model, descriptors, actions, training.LOSS_WEIGHTS)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == set(training.LOSS_WEIGHTS)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_training_protocol_has_no_task_labels_or_calibrated_selection():
    text = " ".join(training.PROTOCOL["excluded"])
    assert "topple" in text and "physical monitor decision" in text
    assert "no task or physical labels" in training.PROTOCOL["checkpoint_selection"]
