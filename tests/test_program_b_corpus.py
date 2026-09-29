from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

pytest.importorskip("mujoco")
import program_b_corpus as pc  # noqa: E402
from jenga_short_held_tails import extract_sim  # noqa: E402


@pytest.fixture(scope="module")
def sim_dir():
    return extract_sim(ROOT / "vendor/panda_express_sim.tar", tempfile.mkdtemp()).parent


def jenga_box():
    return {"kind": "box", "dims": [0.025, 0.020, 0.075]}


def test_jenga_layout_detector():
    upright = np.array([1.0, 0.0, 0.0, 0.0])                      # long (z) axis vertical
    objects = [jenga_box() for _ in range(3)]
    row = np.array([[0.5, -0.04, 0.46], [0.5, 0.0, 0.46], [0.5, 0.04, 0.46]])
    assert pc.is_jenga_layout(objects, row, [upright] * 3)
    lying = pc.quat_multiply(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]), upright)
    assert not pc.is_jenga_layout(objects, row, [lying] * 3)
    triangle = np.array([[0.5, -0.04, 0.46], [0.5, 0.0, 0.46], [0.56, 0.02, 0.46]])
    assert not pc.is_jenga_layout(objects, triangle, [upright] * 3)
    other = [{"kind": "box", "dims": [0.06, 0.06, 0.06]} for _ in range(3)]
    assert not pc.is_jenga_layout(other, row, [upright] * 3)


def test_scene_xml_is_generic():
    scene = pc.sample_scene(np.random.default_rng(0))
    xml = pc.scene_xml(scene)
    assert "block_middle" not in xml and "panda_jenga_setup" not in xml
    assert 2 <= len(scene["objects"]) <= 12
    assert xml.count("<freejoint") == len(scene["objects"])


def test_episode_is_deterministic_and_recorded_every_20ms(sim_dir):
    first = pc.generate_episode(5000, sim_dir, audit=True, steps=6)
    second = pc.generate_episode(5000, sim_dir, audit=True, steps=6)
    assert first["main"]["objects"].shape[0] == 1 + 6 * pc.FRAMES_PER_STEP
    for key in ("objects", "robot", "site", "fingers", "onsets", "audit"):
        assert np.array_equal(first["main"][key], second["main"][key]), key
    assert np.array_equal(first["actions"], second["actions"])
    onsets = first["main"]["onsets"]
    assert np.all(onsets[:, 0] >= 1) and np.all(onsets[:, 1] < pc.TABLE)
    audit = first["main"]["audit"]
    assert np.all(audit[:, 3] <= pc.AUDIT_DISTMAX) and np.all(audit[:, 1] < pc.TABLE)


def test_existing_contacts_at_start_are_not_onsets(sim_dir):
    result = pc.generate_episode(5001, sim_dir, steps=1)
    # Settled objects rest on the table; those contacts exist before recording and must not count.
    table = [row for row in result["main"]["onsets"] if row[2] == pc.TABLE]
    assert len(table) < len(result["scene"]["objects"])


def test_ar1_residual_scale_and_correlation():
    rng = np.random.default_rng(0)
    samples = np.stack([pc.ar1_residual(rng, steps=40) for _ in range(2000)])
    assert abs(samples.std() - pc.RESIDUAL_SIGMA) < 0.1 * pc.RESIDUAL_SIGMA
    lag = np.corrcoef(samples[:, 1:, 0].ravel(), samples[:, :-1, 0].ravel())[0, 1]
    assert abs(lag - pc.RESIDUAL_RHO) < 0.05


def test_splits():
    assert pc.split_of(0) == "train" and pc.split_of(4999) == "train"
    assert pc.split_of(5000) == "validation" and pc.split_of(5199) == "test"
    with pytest.raises(ValueError):
        pc.split_of(5200)
