from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from consequence_monitor import (whole_trajectory_consequence_alarm,
                                 whole_trajectory_separation_curves)
from jenga_d7_explicit_curve import consequence_from_curves, curve_targets, decoded, loss_terms


def test_explicit_curve_loss_balances_five_components():
    target = torch.zeros(2, 6, 41)
    prediction = torch.ones_like(target)
    values = loss_terms(prediction, target)
    assert set(values) == {"early_gap", "full_gap", "curve_action",
                           "curve_early_hold", "curve_late_hold"}
    assert all(torch.isclose(value, torch.tensor(1.)) for value in values.values())


def test_decoded_curves_are_nonnegative():
    values = decoded(torch.tensor([-2., 0., np.log(3.)], dtype=torch.float32))
    assert torch.all(values >= 0)
    assert torch.allclose(values, torch.tensor([0., 0., 2.]), atol=1e-6)


def test_curve_decision_matches_frozen_trace_monitor():
    traces = np.zeros((1, 2, 38, 45), np.float32)
    traces[0, 1, :, 0] = np.linspace(0., 5., 38)
    actions = np.ones((1, 8, 4), np.float32)
    boundary = np.ones(1)
    frozen = whole_trajectory_consequence_alarm(traces, actions, boundary, boundary)
    direct = consequence_from_curves(
        whole_trajectory_separation_curves(traces), actions, boundary, boundary)
    assert direct["alarm"] == frozen.alarm
    assert direct["commitment_pairs"] == frozen.commitment_pairs
    assert direct["persistence_pairs"] == frozen.persistence_pairs
