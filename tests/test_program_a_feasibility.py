from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_reobservation_interface import reanchored_rollout
from program_a_feasibility import (anchored_rollout, covers, deadline, fixed_pairs,
                                   nested_executed, neighbourhood_executed, offset_anchor,
                                   recovery, segment_boundaries, shared_anchor, split_index,
                                   t1_decision)


def additive_rollout(_model, state, actions, _scale):
    values = []
    current = state.clone()
    for action in actions.transpose(0, 1):
        current = current + action[:, :1]
        values.append(current.clone())
    return torch.stack(values, dim=1)


def toy(rows=3, steps=6):
    rng = np.random.default_rng(1)
    starts = rng.normal(size=(rows, 61)).astype(np.float32)
    actions = rng.normal(size=(rows, steps, 4)).astype(np.float32)
    truth = rng.normal(size=(rows, steps, 61)).astype(np.float32)
    return starts, actions, truth


def test_a1_with_every_row_executing_itself_equals_per_branch_oracle():
    starts, actions, truth = toy()
    oracle = reanchored_rollout(None, None, starts, actions, truth, 2, "cpu",
                                rollout_function=additive_rollout)
    a1 = anchored_rollout(None, None, starts, actions, shared_anchor(truth, np.arange(3)),
                          segment_boundaries(2, steps=6), "cpu", rollout_function=additive_rollout)
    assert np.allclose(a1, oracle)


def test_single_segment_is_open_loop():
    starts, actions, truth = toy()
    open_loop = reanchored_rollout(None, None, starts, actions, truth, 6, "cpu",
                                   rollout_function=additive_rollout)
    anchored = anchored_rollout(None, None, starts, actions, shared_anchor(truth, np.zeros(3, int)),
                                [0], "cpu", rollout_function=additive_rollout)
    assert np.allclose(anchored, open_loop)


def test_a1_restarts_every_row_from_the_executed_row():
    starts = np.zeros((2, 61), np.float32)
    actions = np.zeros((2, 4, 4), np.float32); actions[:, :, 0] = [[1.], [2.]]
    truth = np.zeros((2, 4, 61), np.float32); truth[0, :, 0] = [10., 20., 30., 40.]
    result = anchored_rollout(None, None, starts, actions, shared_anchor(truth, np.zeros(2, int)),
                              [0, 2], "cpu", rollout_function=additive_rollout)
    # Row 1 never sees its own truth: it restarts from row 0's observed 20 and adds its own 2s.
    assert np.allclose(result[1, :, 0], [2., 4., 22., 24.])


def test_offset_anchor_keeps_divergence_orthonormalises_and_copies_contacts():
    truth = np.zeros((2, 3, 61), np.float32)
    truth[:, :, 9:27] = np.tile([1., 0., 0., 1., 0., 0.], 3)          # identity columns
    truth[0, 0, 0] = 5.
    predicted = np.zeros_like(truth)
    predicted[:, :, 9:27] = truth[:, :, 9:27]
    predicted[1, 0, 0] = 1.5                                            # row 1 diverged by 1.5
    predicted[1, 0, 9] = 3.                                             # un-normalised rotation
    predicted[1, 0, 45:57] = 1.; predicted[1, 0, 60] = 1.               # its own contacts
    state = offset_anchor(truth, np.zeros(2, int))(1, predicted)
    assert np.isclose(state[1, 0], 6.5)
    blocks = state[1, 9:27].reshape(3, 3, 2)
    assert np.allclose(np.linalg.norm(blocks, axis=1), 1., atol=1e-6)
    assert np.allclose((blocks[:, :, 0] * blocks[:, :, 1]).sum(1), 0., atol=1e-6)
    assert np.all(state[1, 45:57] == 1.) and state[1, 60] == 1.
    assert np.isclose(state[0, 0], 5.)                                  # executed row = its truth


def test_segment_boundaries_with_stop_end_in_one_open_loop_segment():
    assert segment_boundaries(2, steps=8) == [0, 2, 4, 6]
    assert segment_boundaries(2, stop=4, steps=8) == [0, 2, 4]
    assert segment_boundaries(4, stop=0, steps=8) == [0]


def test_executed_index_maps():
    assert list(neighbourhood_executed(2, 3, 1)) == [1, 1, 1, 4, 4, 4]
    # Two groups, three levels, two endpoints: narrow pair is paths 4/5 of each group of six.
    assert list(nested_executed(2, 3, 1)) == [5] * 6 + [11] * 6


def test_split_index_and_deadline():
    pair = np.zeros((2, 10, 61), np.float32)
    pair[1, :, 0] = np.linspace(0., 9., 10)
    scale = np.ones(61)
    assert split_index(pair, scale) == 5               # first value >= 4.5
    assert deadline(5) == 6 and deadline(20) == 8


def test_fixed_pairs_are_extremes_along_principal_directions():
    snippets = np.zeros((5, 1, 2))
    snippets[:, 0, 0] = [-3., -1., 0., 1., 4.]
    snippets[:, 0, 1] = [0., .1, -.1, 0., 0.]
    pairs = fixed_pairs(snippets, 1)
    assert sorted(pairs[0]) == [0, 4]
    assert covers([0, 0, 0, 0, 1], pairs) and not covers([0, 1, 1, 1, 0], pairs)


def test_recovery_undefined_without_headroom():
    assert recovery(.5, .1, .9) == .5
    assert np.isnan(recovery(.5, .5, .5))


def test_t1_prefers_longest_interval_then_a1():
    def summary(passing):
        return {"conditions": {f"{v}_{i}": {"passes": f"{v}_{i}" in passing}
                               for v in ("A1", "A2") for i in (1, 2, 4)}}
    both = {"A1_1", "A2_2", "A1_2"}
    decision = t1_decision({"episode": summary(both), "configuration": summary(both)})
    assert decision["selected"] == {"variant": "A1", "interval": 2}
    one_axis = t1_decision({"episode": summary({"A1_4"}), "configuration": summary(set())})
    assert not one_axis["passes"]
