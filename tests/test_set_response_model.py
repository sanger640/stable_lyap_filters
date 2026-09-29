from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from set_response_model import (DirectMonitorEvidenceModel, DirectSetResponseModel,
                                EventCurveModeRouter,
                                HistoryGroupAmplitudeShapeTemporalCurveModel,
                                InteractionEventGroupAmplitudeShapeCurveModel,
                                JointCurveMonitorEvidenceModel,
                                MixtureGroupAmplitudeShapeCurveModel,
                                NestedPairCurveHead, SnapshotCurveModeRouter,
                                reconstruct_states)


def inputs(batch=2, probes=7, steps=5):
    torch.manual_seed(4)
    state = torch.randn(batch, 61)
    actions = torch.randn(batch, probes, steps, 4)
    norms = (torch.zeros(61), torch.ones(61), torch.zeros(steps, 4), torch.ones(steps, 4))
    return state, actions, norms


def test_set_response_is_permutation_equivariant():
    state, actions, norms = inputs()
    model = DirectSetResponseModel(steps=5, hidden=32, heads=4, layers=2,
                                   set_conditioned=True).eval()
    permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(state, actions, *norms)
        changed = model(state, actions[:, permutation], *norms)[:, inverse]
    assert original.shape == (2, 7, 5, 45)
    assert torch.allclose(original, changed, atol=2e-6, rtol=1e-6)


def test_independent_response_does_not_depend_on_other_probes():
    state, actions, norms = inputs(batch=1)
    model = DirectSetResponseModel(steps=5, hidden=32, heads=4, layers=2,
                                   set_conditioned=False).eval()
    changed = actions.clone(); changed[:, 1:] += 100.
    with torch.no_grad():
        original = model(state, actions, *norms)
        modified = model(state, changed, *norms)
    assert torch.allclose(original[:, 0], modified[:, 0])


def test_reconstruct_states_changes_only_predicted_physical_channels():
    delta = torch.ones(1, 2, 3, 45)
    start = torch.zeros(1, 61)
    template = torch.randn(1, 2, 3, 61)
    scale = torch.arange(1, 46, dtype=torch.float32)
    result = reconstruct_states(delta, start, template, scale)
    assert torch.allclose(result[..., :45], scale)
    assert torch.equal(result[..., 45:], template[..., 45:])


def test_nested_curve_head_is_endpoint_swap_and_level_permutation_equivariant():
    state, actions, norms = inputs(batch=1, probes=6, steps=5)
    pairs = actions.reshape(1, 3, 2, 5, 4)
    backbone = DirectSetResponseModel(steps=5, hidden=32, heads=4, layers=2,
                                      set_conditioned=True)
    model = NestedPairCurveHead(backbone, curve_steps=6, hidden=32).eval()
    with torch.no_grad():
        original = model(state, pairs, *norms)
        swapped = model(state, pairs.flip(2), *norms)
        permutation = torch.tensor([2, 0, 1]); inverse = torch.argsort(permutation)
        reordered = model(state, pairs[:, permutation], *norms)[:, inverse]
    assert original.shape == (1, 3, 8)
    assert torch.allclose(original, swapped, atol=2e-6, rtol=1e-6)
    assert torch.allclose(original, reordered, atol=2e-6, rtol=1e-6)


def test_monitor_evidence_heads_are_endpoint_and_level_invariant():
    state, actions, norms = inputs(batch=2, probes=6, steps=5)
    pairs = actions.reshape(2, 3, 2, 5, 4)
    permutation = torch.tensor([2, 0, 1])
    for model in (
            DirectMonitorEvidenceModel(steps=5, action_horizon=3, curve_steps=6,
                                       hidden=32, heads=4, layers=1),
            JointCurveMonitorEvidenceModel(steps=5, action_horizon=3, curve_steps=6,
                                           hidden=32, heads=4, layers=1)):
        model.eval()
        with torch.no_grad():
            original = model(state, pairs, *norms)
            swapped = model(state, pairs.flip(2), *norms)
            reordered = model(state, pairs[:, permutation], *norms)
        original_evidence = original if torch.is_tensor(original) else original[-1]
        swapped_evidence = swapped if torch.is_tensor(swapped) else swapped[-1]
        reordered_evidence = reordered if torch.is_tensor(reordered) else reordered[-1]
        assert original_evidence.shape == (2, 4)
        assert torch.allclose(original_evidence, swapped_evidence, atol=2e-6, rtol=1e-6)
        assert torch.allclose(original_evidence, reordered_evidence, atol=2e-6, rtol=1e-6)


def test_history_curve_model_preserves_probe_symmetries_and_uses_history():
    torch.manual_seed(7)
    model = HistoryGroupAmplitudeShapeTemporalCurveModel(
        steps=5, action_horizon=3, curve_steps=6, hidden=32, heads=4, layers=2).eval()
    states = torch.randn(2, 4, 61)
    history_actions = torch.randn(2, 3, 4)
    pairs = torch.randn(2, 3, 2, 5, 4)
    norms = (torch.zeros(61), torch.ones(61), torch.zeros(4), torch.ones(4),
             torch.zeros(5, 4), torch.ones(5, 4))
    permutation = torch.tensor([2, 0, 1]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(states, history_actions, pairs, *norms)[0]
        swapped = model(states, history_actions, pairs.flip(2), *norms)[0]
        reordered = model(states, history_actions, pairs[:, permutation], *norms)[0][:, inverse]
        changed_states = states.clone(); changed_states[:, 0] += 2.
        changed = model(changed_states, history_actions, pairs, *norms)[0]
    assert original.shape == (2, 3, 8)
    assert torch.allclose(original, swapped, atol=2e-6, rtol=1e-6)
    assert torch.allclose(original, reordered, atol=2e-6, rtol=1e-6)
    assert not torch.allclose(original, changed)


def test_interaction_event_features_expose_contact_creation_and_loss():
    model = InteractionEventGroupAmplitudeShapeCurveModel(
        steps=5, action_horizon=3, curve_steps=6, hidden=32, heads=4, layers=1)
    states = torch.zeros(1, 4, 61); actions = torch.zeros(1, 3, 4)
    states[0, 1, 45] = 1.; states[0, 2, 45] = 1.
    features = model.event_features(
        states, actions, torch.zeros(45), torch.ones(45), torch.zeros(4), torch.ones(4))
    contact_offset = 2 * 45
    created_offset = contact_offset + 2 * 12
    lost_offset = created_offset + 12
    assert features[0, 0, created_offset] == 1.
    assert features[0, 2, lost_offset] == 1.
    assert features.shape[-1] == model.EVENT_DIM


def test_interaction_event_model_preserves_probe_symmetries():
    torch.manual_seed(8)
    model = InteractionEventGroupAmplitudeShapeCurveModel(
        steps=5, action_horizon=3, curve_steps=6, hidden=32, heads=4, layers=2).eval()
    states = torch.randn(2, 4, 61); states[..., 45:57] = (states[..., 45:57] > 0).float()
    prior = torch.randn(2, 3, 4); pairs = torch.randn(2, 3, 2, 5, 4)
    norms = (torch.zeros(61), torch.ones(61), torch.zeros(45), torch.ones(45),
             torch.zeros(4), torch.ones(4), torch.zeros(5, 4), torch.ones(5, 4))
    permutation = torch.tensor([2, 0, 1]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(states, prior, pairs, *norms)[0]
        swapped = model(states, prior, pairs.flip(2), *norms)[0]
        reordered = model(states, prior, pairs[:, permutation], *norms)[0][:, inverse]
    assert torch.allclose(original, swapped, atol=2e-6, rtol=1e-6)
    assert torch.allclose(original, reordered, atol=2e-6, rtol=1e-6)


def test_curve_mixture_uses_one_mode_distribution_for_complete_group():
    torch.manual_seed(9)
    state, actions, norms = inputs(batch=2, probes=6, steps=5)
    pairs = actions.reshape(2, 3, 2, 5, 4)
    model = MixtureGroupAmplitudeShapeCurveModel(
        modes=3, steps=5, action_horizon=3, curve_steps=6,
        hidden=32, heads=4, layers=2).eval()
    permutation = torch.tensor([2, 0, 1]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        curves, logits, amplitude = model(state, pairs, *norms)
        swapped = model(state, pairs.flip(2), *norms)
        reordered = model(state, pairs[:, permutation], *norms)
    assert curves.shape == (2, 3, 3, 8)
    assert logits.shape == amplitude.shape == (2, 3)
    assert torch.all(curves >= 0)
    assert torch.allclose(curves, swapped[0], atol=2e-6, rtol=1e-6)
    assert torch.allclose(logits, swapped[1], atol=2e-6, rtol=1e-6)
    assert torch.allclose(curves, reordered[0][:, :, inverse], atol=2e-6, rtol=1e-6)
    assert torch.allclose(logits, reordered[1], atol=2e-6, rtol=1e-6)


def test_curve_mode_routers_are_level_permutation_invariant():
    torch.manual_seed(10)
    pair_features = torch.randn(4, 6, 32)
    events = torch.randn(4, 3, 142)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    snapshot = SnapshotCurveModeRouter(hidden=32, modes=3).eval()
    event = EventCurveModeRouter(hidden=32, modes=3).eval()
    with torch.no_grad():
        assert torch.allclose(snapshot(pair_features), snapshot(pair_features[:, permutation]))
        assert torch.allclose(event(pair_features, events),
                              event(pair_features[:, permutation], events))
