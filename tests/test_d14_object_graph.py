from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d14_object_graph import permute_objects
from set_response_model import ObjectGraphJointMonitorEvidenceModel


def inputs(batch=2):
    generator = torch.Generator().manual_seed(14)
    state = torch.randn(batch, 61, generator=generator) * .02
    state[:, 45:57] = (torch.rand(batch, 12, generator=generator) > .5).float()
    actions = torch.randn(batch, 6, 2, 38, 4, generator=generator) * .01
    state_mean = torch.zeros(61); state_scale = torch.ones(61)
    action_mean = torch.zeros(38, 4); action_scale = torch.ones(38, 4)
    return state, actions, state_mean, state_scale, action_mean, action_scale


def model():
    return ObjectGraphJointMonitorEvidenceModel(
        hidden=32, heads=4, layers=1, graph_rounds=2, action_horizon=8, curve_steps=39).eval()


def test_graph_joint_output_contract_and_gradients():
    values = inputs(); network = model().train()
    curve, amplitude, shape, evidence = network(*values)
    assert curve.shape == (2, 6, 41)
    assert amplitude.shape == (2, 1, 1)
    assert shape.shape == (2, 6, 41)
    assert evidence.shape == (2, 4)
    (curve.mean() + evidence.mean()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in network.parameters())


def test_object_relabelling_is_exactly_invariant():
    torch.manual_seed(1); network = model(); values = inputs()
    with torch.no_grad():
        original = network(*values)
        relabelled = network(permute_objects(values[0], (2, 0, 1)), *values[1:])
    for first, second in zip(original, relabelled):
        assert torch.allclose(first, second, atol=2e-6)


def test_joint_horizontal_translation_is_exactly_invariant():
    torch.manual_seed(2); network = model(); values = list(inputs())
    translated_state = values[0].clone(); translated_actions = values[1].clone()
    offset = torch.tensor([.08, -.035])
    translated_state[:, :9].reshape(2, 3, 3)[..., :2] += offset
    translated_state[:, 57:59] += offset
    translated_actions[..., :2] += offset
    with torch.no_grad():
        original = network(*values)
        translated = network(translated_state, translated_actions, *values[2:])
    for first, second in zip(original, translated):
        assert torch.allclose(first, second, atol=2e-6)


def test_endpoint_and_level_symmetries_remain_intact():
    torch.manual_seed(3); network = model(); values = inputs(); order = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        original = network(*values)
        endpoint = network(values[0], values[1].flip(2), *values[2:])
        levels = network(values[0], values[1][:, order], *values[2:])
    inverse = torch.argsort(order)
    for first, second in zip(original, endpoint):
        assert torch.allclose(first, second, atol=2e-6)
    for first, second in zip(original, levels):
        if second.ndim >= 3 and second.shape[1] == 6:
            second = second[:, inverse]
        assert torch.allclose(first, second, atol=2e-6)
