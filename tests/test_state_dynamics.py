import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from state_dynamics import (STATE_DIM, StepGraphNet, apply_step, contact_targets,  # noqa: E402
                            delta_targets, rollout)


def random_state(batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, STATE_DIM, generator=g) * 0.01
    state[:, 45:57] = (torch.rand(batch, 12, generator=g) > 0.5).float()
    state[:, 60] = 1.0
    return state


def unit_scale():
    return torch.ones(15), torch.ones(3)


def test_step_output_shapes():
    model = StepGraphNet(hidden=32, rounds=2)
    out = model(random_state(), torch.zeros(5, 4))
    assert out["block_mean"].shape == (5, 3, 15)
    assert out["block_contact_logits"].shape == (5, 3, 3)
    assert out["gripper_mean"].shape == (5, 3)
    assert out["pair_contact_logits"].shape == (5, 3)


def test_swapping_the_two_neighbours_swaps_their_predictions():
    torch.manual_seed(0)
    model = StepGraphNet(hidden=32, rounds=2).eval()
    state = random_state(batch=1)
    swapped = state.clone()
    # Swap left (block 1) and right (block 2) everywhere they appear.
    for base, width in ((0, 3), (9, 6), (27, 6)):
        a, b = slice(base + width, base + 2 * width), slice(base + 2 * width, base + 3 * width)
        swapped[:, a], swapped[:, b] = state[:, b].clone(), state[:, a].clone()
    c = state[:, 45:57]
    sc = swapped[:, 45:57]
    sc[:, 0], sc[:, 1] = c[:, 1], c[:, 0]                  # middle-left <-> middle-right
    for offset in (3, 6, 9):                               # table / floor / robot per block
        sc[:, offset + 1], sc[:, offset + 2] = c[:, offset + 2], c[:, offset + 1]
    action = torch.tensor([[0.001, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        a, b = model(state, action), model(swapped, action)
    assert torch.allclose(a["block_mean"][0, 1], b["block_mean"][0, 2], atol=1e-5)
    assert torch.allclose(a["block_mean"][0, 0], b["block_mean"][0, 0], atol=1e-5)


def test_integration_adds_the_predicted_change():
    state = random_state(batch=2)
    out = {"block_mean": torch.ones(2, 3, 15), "block_logvar": torch.zeros(2, 3, 15),
           "block_contact_logits": torch.full((2, 3, 3), 10.0),
           "gripper_mean": torch.ones(2, 3), "gripper_logvar": torch.zeros(2, 3),
           "pair_contact_logits": torch.full((2, 3), -10.0)}
    new = apply_step(state, torch.zeros(2, 4), out, unit_scale())
    assert torch.allclose(new[:, 0:45], state[:, 0:45] + 1)
    assert torch.allclose(new[:, 57:60], state[:, 57:60] + 1)
    # Contacts live at 45:57: pairs first (45:48), then per-block table/floor/robot (48:57).
    assert torch.all(new[:, 48:57] > 0.99) and torch.all(new[:, 45:48] < 0.01)


def test_true_deltas_round_trip_through_integration():
    state, nxt = random_state(seed=1), random_state(seed=2)
    blocks, grip = delta_targets(state, nxt)
    per_block, pairs = contact_targets(nxt)
    logit = lambda p: torch.where(p > 0.5, torch.tensor(20.0), torch.tensor(-20.0))  # noqa: E731
    out = {"block_mean": blocks, "block_logvar": torch.zeros_like(blocks),
           "block_contact_logits": logit(per_block), "gripper_mean": grip,
           "gripper_logvar": torch.zeros_like(grip), "pair_contact_logits": logit(pairs)}
    rebuilt = apply_step(state, torch.zeros(5, 4), out, unit_scale())
    assert torch.allclose(rebuilt[:, :60], nxt[:, :60], atol=1e-6)


def test_gripper_closes_on_the_close_command():
    state = random_state(batch=1); state[:, 60] = 0.0
    out = StepGraphNet(hidden=16, rounds=1)(state, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    new = apply_step(state, torch.tensor([[0.0, 0.0, 0.0, 1.0]]), out, unit_scale())
    assert new[0, 60] == 1.0


def test_rollout_length_and_gradients():
    model = StepGraphNet(hidden=16, rounds=1)
    states = rollout(model, random_state(batch=3), torch.zeros(3, 7, 4), unit_scale())
    assert states.shape == (3, 7, STATE_DIM)
    states.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_neighbour_tilt_reads_row_major_rotation():
    import numpy as np
    sys.path.insert(0, str(ROOT / "eval"))
    from jenga_w5_eval import neighbour_tilt_deg
    upright = np.eye(3)[:, :2].reshape(-1)                   # row-major first two columns
    tipped = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)[:, :2].reshape(-1)  # 90 deg
    state = np.zeros(61)
    state[9:15] = upright; state[15:21] = upright; state[21:27] = tipped
    tilt = neighbour_tilt_deg(state[None])[0]
    assert abs(tilt[0]) < 1e-6 and abs(tilt[1] - 90) < 1e-6
