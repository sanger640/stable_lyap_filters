from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"eval")); sys.path.insert(0, str(ROOT/"src"))

from panda_push_patch_correspondence import (correspondence_positions,
                                             geometric_trajectories,
                                             patch_coordinates)


def test_identity_descriptors_stay_at_anchor_coordinates():
    start = np.eye(4, dtype=np.float32)
    frames = np.stack([start, start])
    positions = correspondence_positions(start, frames, top_k=1)
    assert np.allclose(positions[0], patch_coordinates(4))
    assert np.allclose(positions[1], patch_coordinates(4))


def test_permuted_descriptors_follow_their_new_geometric_positions():
    start = np.eye(4, dtype=np.float32)
    # current[target] = start[source]; source 0 therefore moves to target 1, etc.
    permutation = np.asarray([1, 0, 3, 2])
    current = start[permutation]
    positions = correspondence_positions(start, current[None], top_k=1)[0]
    inverse = np.argsort(permutation)
    assert np.allclose(positions, patch_coordinates(4)[inverse])


def test_static_correspondence_has_zero_velocity():
    start = np.eye(4, dtype=np.float32)
    frames = np.broadcast_to(start, (2, 3, 4, 4))
    geometry = geometric_trajectories(start, frames, [1, 2, 4])
    reshaped = geometry.reshape(2, 3, 4, 4)
    assert np.allclose(reshaped[..., 2:], 0.)
