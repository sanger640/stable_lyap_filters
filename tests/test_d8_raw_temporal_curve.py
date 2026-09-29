from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d8_raw_temporal_curve import frozen_gate
from set_response_model import DirectTemporalPairCurveModel


def test_direct_temporal_curve_model_is_pair_and_level_equivariant():
    torch.manual_seed(8)
    model = DirectTemporalPairCurveModel(steps=10, action_horizon=4, curve_steps=11,
                                         hidden=32, heads=4, layers=2).eval()
    state = torch.randn(2, 61); actions = torch.randn(2, 5, 2, 10, 4)
    norms = (torch.zeros(61), torch.ones(61), torch.zeros(10, 4), torch.ones(10, 4))
    permutation = torch.tensor([3, 0, 4, 1, 2]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(state, actions, *norms)
        swapped = model(state, actions.flip(2), *norms)
        reordered = model(state, actions[:, permutation], *norms)[:, inverse]
    assert original.shape == (2, 5, 13)
    assert torch.allclose(original, swapped, atol=2e-6, rtol=1e-6)
    assert torch.allclose(original, reordered, atol=2e-6, rtol=1e-6)


def test_frozen_gate_requires_exact_nested_decisions():
    errors = {name: 0. for name in ("early_gap", "full_gap", "curve_action",
                                     "curve_early_hold", "curve_late_hold")}
    nested = {"boundary": {"agreement": 1.}, "alarm": {"agreement": 1.}}
    checks = {"endpoint_swap_max_abs": 0., "level_permutation_max_abs": 0.}
    assert frozen_gate(errors, nested, checks)["passes"]
    nested["alarm"]["agreement"] = .875
    assert not frozen_gate(errors, nested, checks)["passes"]
