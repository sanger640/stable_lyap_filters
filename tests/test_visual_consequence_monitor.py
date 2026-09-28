import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"src"))

from visual_consequence_monitor import (visual_whole_trajectory_consequence_alarm,
                                        visual_whole_trajectory_separation_curves)


def visual_pairs(kind, dimensions=11, steps=38):
    pairs = np.zeros((3, 2, steps, dimensions), float)
    time = np.arange(1, steps+1, dtype=float)
    exposure = np.minimum(time, 8.)/8.
    for pair, amplitude in enumerate((.8, 1., 1.2)):
        if kind == "passive":
            response = amplitude*exposure
        elif kind == "commit":
            response = amplitude*exposure + 6*amplitude*(
                1-np.exp(-np.maximum(time-4., 0.)/3.))
        else:
            raise ValueError(kind)
        direction = np.linspace(.2, 1.2, dimensions)
        pairs[pair, 1] = response[:, None]*direction[None]
    return pairs


def decision(kind, boundary=True):
    action = np.ones((3, 8, 3)); action[:, :, 1:] = 0.
    evidence = np.ones(3) if boundary else -np.ones(3)
    return visual_whole_trajectory_consequence_alarm(
        visual_pairs(kind), action, evidence, evidence)


def test_visual_curve_is_invariant_to_scale_and_feature_permutation():
    pairs = visual_pairs("commit")
    reference = visual_whole_trajectory_separation_curves(pairs)
    assert np.allclose(reference, visual_whole_trajectory_separation_curves(1000*pairs))
    assert np.allclose(reference, visual_whole_trajectory_separation_curves(pairs[..., ::-1]))


def test_visual_native_rejects_smooth_forced_response():
    assert not decision("passive").alarm


def test_visual_native_accepts_committed_persistent_response():
    result = decision("commit")
    assert result.alarm
    assert result.consequential_pairs == 3


def test_visual_native_still_requires_boundary_evidence():
    assert not decision("commit", boundary=False).alarm
