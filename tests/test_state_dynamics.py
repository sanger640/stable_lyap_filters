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
    from state_dynamics import neighbour_tilt_deg
    upright = np.eye(3)[:, :2].reshape(-1)                   # row-major first two columns
    tipped = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)[:, :2].reshape(-1)  # 90 deg
    state = np.zeros(61)
    state[9:15] = upright; state[15:21] = upright; state[21:27] = tipped
    tilt = neighbour_tilt_deg(state[None])[0]
    assert abs(tilt[0]) < 1e-6 and abs(tilt[1] - 90) < 1e-6


from state_dynamics import StepGraphMoE, load_balance  # noqa: E402


def test_moe_output_matches_the_baseline_contract():
    model = StepGraphMoE(hidden=32, rounds=2, experts=3)
    out = model(random_state(), torch.zeros(5, 4))
    assert out["block_mean"].shape == (5, 3, 15) and out["block_logvar"].shape == (5, 3, 15)
    assert out["block_contact_logits"].shape == (5, 3, 3)
    assert out["gate_probabilities"].shape == (5, 3, 3)
    states = rollout(model, random_state(batch=2), torch.zeros(2, 4, 4), unit_scale())
    assert states.shape == (2, 4, STATE_DIM)


def test_moe_switches_hard_and_trains_the_gate():
    torch.manual_seed(0)
    model = StepGraphMoE(hidden=32, rounds=2, experts=3)
    model.train()
    out = model(random_state(), torch.zeros(5, 4))
    assert torch.all(out["regime"].sum(-1) == 1)                       # exactly one expert each
    (out["block_mean"].sum() + load_balance(out["gate_probabilities"])).backward()
    assert model.gate[0].weight.grad is not None and model.gate[0].weight.grad.abs().sum() > 0


def test_moe_eval_is_deterministic_and_equivariant():
    torch.manual_seed(0)
    model = StepGraphMoE(hidden=32, rounds=2, experts=3).eval()
    state = random_state(batch=1)
    swapped = state.clone()
    for base, width in ((0, 3), (9, 6), (27, 6)):
        a, b = slice(base + width, base + 2 * width), slice(base + 2 * width, base + 3 * width)
        swapped[:, a], swapped[:, b] = state[:, b].clone(), state[:, a].clone()
    c, sc = state[:, 45:57], swapped[:, 45:57]
    sc[:, 0], sc[:, 1] = c[:, 1], c[:, 0]
    for offset in (3, 6, 9):
        sc[:, offset + 1], sc[:, offset + 2] = c[:, offset + 2], c[:, offset + 1]
    action = torch.tensor([[0.001, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        a1, a2, b = model(state, action), model(state, action), model(swapped, action)
    assert torch.equal(a1["block_mean"], a2["block_mean"])
    assert torch.allclose(a1["block_mean"][0, 1], b["block_mean"][0, 2], atol=1e-5)


def test_load_balance_is_zero_when_usage_is_uniform():
    assert abs(float(load_balance(torch.full((4, 3, 3), 1 / 3)))) < 1e-6
    assert float(load_balance(torch.tensor([[[1.0, 0.0, 0.0]]]).repeat(4, 3, 1))) > 1.0


from state_dynamics import (StepSwitchingEdgeGNN, initialise_switching_from_single,  # noqa: E402
                            switching_mode_regularizers)


def test_switching_edge_model_contract_and_persistent_state():
    model = StepSwitchingEdgeGNN(hidden=32, rounds=2, modes=3).eval()
    state, action = random_state(), torch.zeros(5, 4)
    first = model(state, action)
    assert first["block_mean"].shape == (5, 3, 15)
    assert first["gate_probabilities"].shape == (5, 12, 3)
    assert first["next_mode_state"].shape == (5, 12, 3)
    assert torch.all(first["regime"].sum(-1) == 1)
    second = model(state, action, first["next_mode_state"])
    assert second["next_mode_state"].shape == first["next_mode_state"].shape
    gates = []
    states = rollout(model, random_state(batch=2), torch.zeros(2, 4, 4), unit_scale(),
                     collect=gates)
    assert states.shape == (2, 4, STATE_DIM)
    assert torch.stack(gates, dim=1).shape == (2, 4, 12, 3)


def test_switching_edge_gate_and_message_experts_receive_gradients():
    torch.manual_seed(0)
    model = StepSwitchingEdgeGNN(hidden=24, rounds=2, modes=3).train()
    gates = []
    states = rollout(model, random_state(batch=3), torch.zeros(3, 4, 4), unit_scale(),
                     collect=gates)
    regularizers = switching_mode_regularizers(torch.stack(gates, dim=1))
    loss = states.square().mean() + 0.01 * sum(regularizers.values())
    loss.backward()
    assert model.mode_gate[0].weight.grad is not None
    assert model.mode_gate[0].weight.grad.abs().sum() > 0
    expert_gradients = [parameter.grad for experts in model.mode_edge_updates for expert in experts
                        for parameter in expert.parameters()]
    assert any(gradient is not None and gradient.abs().sum() > 0
               for gradient in expert_gradients)


def test_switching_edge_eval_is_deterministic_and_equivariant():
    torch.manual_seed(0)
    model = StepSwitchingEdgeGNN(hidden=32, rounds=2, modes=3).eval()
    state = random_state(batch=1)
    swapped = state.clone()
    for base, width in ((0, 3), (9, 6), (27, 6)):
        left = slice(base + width, base + 2 * width)
        right = slice(base + 2 * width, base + 3 * width)
        swapped[:, left], swapped[:, right] = state[:, right].clone(), state[:, left].clone()
    contacts, swapped_contacts = state[:, 45:57], swapped[:, 45:57]
    swapped_contacts[:, 0], swapped_contacts[:, 1] = contacts[:, 1], contacts[:, 0]
    for offset in (3, 6, 9):
        swapped_contacts[:, offset + 1], swapped_contacts[:, offset + 2] = (
            contacts[:, offset + 2], contacts[:, offset + 1])
    action = torch.tensor([[0.001, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        first = model(state, action)
        repeated = model(state, action)
        permuted = model(swapped, action)
    assert torch.equal(first["block_mean"], repeated["block_mean"])
    assert torch.allclose(first["block_mean"][0, 1], permuted["block_mean"][0, 2], atol=1e-5)


def test_switching_regularizers_have_expected_extrema():
    uniform = torch.full((2, 4, 12, 3), 1 / 3)
    terms = switching_mode_regularizers(uniform)
    assert abs(float(terms["mode_balance"])) < 1e-6
    assert abs(float(terms["mode_entropy"]) - 1.0) < 1e-6
    assert abs(float(terms["mode_persistence"])) < 1e-6
    collapsed = torch.zeros_like(uniform); collapsed[..., 0] = 1
    assert float(switching_mode_regularizers(collapsed)["mode_balance"]) > 1.0
    alternating = collapsed.clone()
    alternating[:, 1::2, :, 0] = 0; alternating[:, 1::2, :, 1] = 1
    assert float(switching_mode_regularizers(alternating)["mode_persistence"]) > 0


def test_switching_checkpoint_loads_through_shared_evaluator(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    from jenga_w5_eval import load_model
    model = StepSwitchingEdgeGNN(24, 2, 3)
    path = tmp_path / "edge_switch.pt"
    torch.save({"model": model.state_dict(), "hidden": 24, "rounds": 2,
                "kind": "edge_switch", "experts": 3, "modes": 3, "substeps": 1,
                "block_scale": torch.ones(15), "grip_scale": torch.ones(3)}, path)
    loaded, scale = load_model(str(path), "cpu")
    assert isinstance(loaded, StepSwitchingEdgeGNN)
    assert loaded.modes == 3 and scale[0].shape == (15,)


def test_switching_warm_start_exactly_embeds_single_graph_dynamics():
    torch.manual_seed(4)
    single = StepGraphNet(32, 2).eval()
    switching = StepSwitchingEdgeGNN(32, 2, 3).eval()
    copied = initialise_switching_from_single(switching, single.state_dict())
    assert any(name.startswith("mode_edge_updates.0.2") for name in copied)
    state, action = random_state(batch=4), torch.randn(4, 4) * 0.01
    with torch.no_grad():
        expected, actual = single(state, action), switching(state, action)
    for name in ("block_mean", "block_logvar", "block_contact_logits", "gripper_mean",
                 "gripper_logvar", "pair_contact_logits"):
        assert torch.equal(expected[name], actual[name])


def test_baseline_checkpoint_still_loads_after_refactor():
    import os
    path = ROOT / "results/jenga/w5_stepnet.pt"
    if not os.path.exists(path):
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = StepGraphNet(state["hidden"], state["rounds"])
    model.load_state_dict(state["model"])                              # strict: names unchanged


def test_substep_recording_matches_the_simulators_own_execute():
    import os
    import tempfile
    import numpy as np
    archive = ROOT / "vendor/panda_express_sim.tar"
    if not os.path.exists(archive):
        return
    sys.path.insert(0, str(ROOT / "eval"))
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import execute_recording, step_state
    with tempfile.TemporaryDirectory() as temp:
        sim = DirectJengaSim(str(extract_sim(archive, temp)))
        try:
            sim.reset(7)
            action = np.array([sim.target[0] + 0.01, sim.target[1], sim.target[2] - 0.005, 0.0],
                              np.float32)
            snap = sim.snapshot()
            sim.execute(action)
            reference = step_state(sim)
            sim.restore(snap)
            readings = execute_recording(sim, action, every=10)
            assert len(readings) == sim.steps_per_action // 10
            assert np.array_equal(readings[-1], reference)          # bit-identical final state
        finally:
            sim.close()


from state_dynamics import TransitionWeights, change_magnitude  # noqa: E402


def test_transition_weights_favour_rare_large_changes():
    torch.manual_seed(0)
    still = torch.rand(9800) * 1e-3 + 1e-4          # the common case: almost nothing moves
    large = torch.rand(200) * 1.0 + 1.0             # rare large changes
    m = torch.cat([still, large])
    w = TransitionWeights(m)
    assert abs(float(w(m).mean()) - 1.0) < 0.05                 # normalised to mean 1
    # Rare large changes are clearly up-weighted (about 4x for this 98/2 split).
    assert float(w(large).mean()) > 2 * float(w(still).mean())
    assert float(w(m).max()) <= TransitionWeights.CAP


def test_change_magnitude_ignores_contacts_and_gripper():
    a = torch.zeros(1, 61); b = a.clone()
    b[0, 45:61] = 5.0                                # contacts and gripper change only
    assert float(change_magnitude(a, b, torch.ones(61))) == 0.0
    b[0, 0] = 3.0; b[0, 1] = 4.0
    assert abs(float(change_magnitude(a, b, torch.ones(61))) - 5.0) < 1e-6


def test_transition_weights_never_boost_rare_tiny_changes():
    torch.manual_seed(0)
    jitter = torch.rand(50) * 1e-9 + 1e-10           # rare, but TINY: must not be up-weighted
    still = torch.rand(9750) * 1e-3 + 1e-4
    large = torch.rand(200) * 1.0 + 1.0
    w = TransitionWeights(torch.cat([jitter, still, large]))
    assert float(w(jitter).max()) <= float(w(still).max()) + 1e-6
    assert float(w(large).mean()) > 2 * float(w(still).mean())


def test_orthonormalise_rotation_restores_a_drifted_frame():
    """Integrating additive deltas lets the stored columns drift; the projection undoes that."""
    import numpy as np
    from state_dynamics import N_BLOCKS, neighbour_tilt_deg, orthonormalise_rotation
    torch.manual_seed(0)
    # A valid rotation per block, stored ROW-major as the first two columns.
    rot = []
    for _ in range(N_BLOCKS):
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        rot.append(q[:, :2].reshape(-1).float())
    clean = torch.cat(rot)[None]
    drifted = clean + 0.05 * torch.randn_like(clean)
    fixed = orthonormalise_rotation(drifted)

    columns = fixed.view(1, N_BLOCKS, 3, 2)
    c0, c1 = columns[..., 0], columns[..., 1]
    assert torch.allclose(c0.norm(dim=-1), torch.ones(1, N_BLOCKS), atol=1e-5)
    assert torch.allclose(c1.norm(dim=-1), torch.ones(1, N_BLOCKS), atol=1e-5)
    assert torch.allclose((c0 * c1).sum(-1), torch.zeros(1, N_BLOCKS), atol=1e-5)
    # An already-valid frame is left alone, so the tilt read from it does not move.
    assert torch.allclose(orthonormalise_rotation(clean), clean, atol=1e-5)
    state = np.zeros((1, 61), np.float32)
    state[0, 9:27] = clean[0].numpy()
    before = neighbour_tilt_deg(state)
    state[0, 9:27] = orthonormalise_rotation(clean)[0].numpy()
    assert np.allclose(before, neighbour_tilt_deg(state), atol=1e-4)


def test_hard_contacts_round_and_soft_contacts_do_not():
    from state_dynamics import CONTACT, StepGraphNet, apply_step
    torch.manual_seed(0)
    model = StepGraphNet(32, 1)
    state = torch.randn(4, 61)
    state[:, CONTACT] = torch.rand(4, 12)
    action = torch.randn(4, 4)
    out = model(state, action)
    scale = (torch.ones(15), torch.ones(3))
    soft = apply_step(state, action, out, scale)[:, CONTACT]
    hard = apply_step(state, action, out, scale, hard_contacts=True)[:, CONTACT]
    assert set(hard.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(hard, (soft > 0.5).to(soft.dtype))
    assert not torch.equal(soft, hard)


def test_mlp_matches_the_graph_net_contract():
    """The flat ablation must be a drop-in: same output keys and shapes, same integration."""
    from state_dynamics import StepGraphNet, StepMLP, apply_step
    torch.manual_seed(0)
    state, action = torch.randn(5, 61), torch.randn(5, 4)
    graph, flat = StepGraphNet(32, 1), StepMLP(64, 2)
    a, b = graph(state, action), flat(state, action)
    assert set(a) <= set(b) | {"gate_probabilities", "regime"}
    for key in ("block_mean", "block_logvar", "block_contact_logits", "gripper_mean",
                "gripper_logvar", "pair_contact_logits"):
        assert a[key].shape == b[key].shape, key
    scale = (torch.ones(15), torch.ones(3))
    assert apply_step(state, action, b, scale).shape == state.shape


def test_load_model_restores_the_integration_flags(tmp_path):
    """A checkpoint must roll out the way it was trained, not the way the defaults say."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from jenga_w5_eval import load_model
    from state_dynamics import StepGraphNet
    model = StepGraphNet(32, 1)
    path = tmp_path / "check.pt"
    torch.save({"model": model.state_dict(), "hidden": 32, "rounds": 1, "kind": "single",
                "experts": 1, "substeps": 1, "orthonormalise": True, "hard_contacts": True,
                "block_scale": torch.ones(15), "grip_scale": torch.ones(3)}, path)
    loaded, _ = load_model(str(path), "cpu")
    assert loaded.orthonormalise is True
    assert loaded.hard_contacts is True
