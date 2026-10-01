from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

pytest.importorskip("mujoco")
import program_b_reference as pr  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    xml = extract_sim(ROOT / "vendor/panda_express_sim.tar", tempfile.mkdtemp())
    s = DirectJengaSim(str(xml))
    s.reset(100)
    yield s
    s.close()


def block_qpos(sim):
    from jenga_short_held_tails import BLOCK_NAMES
    m = sim.model
    return np.stack([sim.data.qpos[m.jnt_qposadr[m.body_jntadr[m.body(n).id]]:][:7]
                     for n in BLOCK_NAMES])


def test_noise_is_one_deterministic_draw_per_state_and_zero_at_sigma_zero():
    a = pr.pose_noise((0, 5), 0.00265)
    b = pr.pose_noise((0, 5), 0.00265)
    c = pr.pose_noise((0, 6), 0.00265)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])
    zero = pr.pose_noise((0, 5), 0.0)
    assert not zero[0].any() and not zero[1].any()


def test_perturbation_moves_blocks_by_the_draw_and_keeps_unit_quaternions(sim):
    snapshot = sim.snapshot()
    before = block_qpos(sim)
    position, rotation = pr.pose_noise((0, 1), 0.00265)
    pr.perturb_blocks(sim, position, rotation)
    after = block_qpos(sim)
    assert np.allclose(after[:, :3] - before[:, :3], position)
    assert np.allclose(np.linalg.norm(after[:, 3:], axis=1), 1.0)
    sim.restore(snapshot)
    pr.perturb_blocks(sim, np.zeros((3, 3)), np.zeros((3, 3)))
    assert np.allclose(block_qpos(sim), before)
    sim.restore(snapshot)


def test_friction_scales_only_blocks_and_is_restored(sim):
    base = sim.model.geom_friction[:, 0].copy()
    with pr.Friction(sim, 1.1) as friction:
        scaled = sim.model.geom_friction[:, 0].copy()
        others = np.setdiff1d(np.arange(len(base)), friction.geoms)
        assert np.allclose(scaled[friction.geoms], base[friction.geoms] * 1.1)
        assert np.array_equal(scaled[others], base[others])
    assert np.array_equal(sim.model.geom_friction[:, 0], base)


def test_conditions_and_cousin_factors():
    names = [c[0] for c in pr.conditions()]
    assert names[:4] == ["sigma_0", "sigma_1", "sigma_2.65", "sigma_4.44"] and len(names) == 8
    factors = pr.friction_factors()
    assert factors[0] == 1.0 and np.all((factors[1:] >= 0.9) & (factors[1:] <= 1.1))
