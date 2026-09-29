from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d14_object_graph import permute_objects
from jenga_d16_history_separability import (history_context_distance,
                                             history_control_vectors,
                                             transition_vectors)
from set_response_model import ObjectGraphJointMonitorEvidenceModel


def history(batch=3):
    generator = torch.Generator().manual_seed(16)
    states = torch.randn(batch, 4, 61, generator=generator) * .02
    states[..., 45:57] = (torch.rand(batch, 4, 12, generator=generator) > .5).float()
    controls = torch.randn(batch, 3, 4, generator=generator) * .01
    scale = np.ones((38, 4), np.float32)
    return states.numpy(), controls.numpy(), scale


def test_history_controls_are_joint_translation_invariant():
    states, controls, scale = history()
    translated_states = states.copy(); translated_controls = controls.copy()
    offset = np.asarray([.07, -.04, .02])
    translated_states[..., :9].reshape(3, 4, 3, 3)[..., :] += offset
    translated_states[..., 57:60] += offset
    translated_controls[..., :3] += offset
    assert np.allclose(history_control_vectors(states, controls, scale),
                       history_control_vectors(translated_states, translated_controls, scale),
                       atol=1e-7)


def test_history_context_distance_minimizes_one_consistent_object_permutation():
    states, controls, scale = history()
    permuted = permute_objects(
        torch.as_tensor(states).reshape(-1, 61), (2, 0, 1)).reshape_as(
            torch.as_tensor(states)).numpy()
    distance = history_context_distance(states, controls, permuted, controls, scale, "cpu")
    assert np.all(np.min(distance, axis=1) < 1e-6)


def test_transition_vector_has_three_graph_deltas_and_three_controls():
    states, controls, scale = history(batch=2)
    values = transition_vectors(ObjectGraphJointMonitorEvidenceModel,
                                states, controls, scale, "cpu")
    # D14 graph vector is 3*22 + 6*5 + 5 = 101; three deltas plus 3*4 controls.
    assert values.shape == (2, 3 * 101 + 12)
