from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d17_enhanced_state_data import (execute_with_impulse, instantaneous_contacts,
                                            robot_proprioception)
from jenga_d17_enhanced_separability import (contact_normalizers, feature_normalize,
                                               normalize_contacts, permute_contacts)


def simulator_pair():
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    temporary = tempfile.TemporaryDirectory()
    xml = str(extract_sim(ROOT / "vendor/panda_express_sim.tar", temporary.name))
    return DirectJengaSim(xml), DirectJengaSim(xml), temporary


def test_enhanced_channels_have_frozen_shapes_and_are_finite():
    first, second, temporary = simulator_pair()
    try:
        first.reset(100)
        assert robot_proprioception(first).shape == (24,)
        assert instantaneous_contacts(first).shape == (12, 5)
        assert np.all(np.isfinite(robot_proprioception(first)))
        assert np.all(np.isfinite(instantaneous_contacts(first)))
    finally:
        first.close(); second.close(); temporary.cleanup()


def test_force_recording_is_state_identical_to_normal_execute():
    first, second, temporary = simulator_pair()
    try:
        first.reset(101); second.reset(101)
        action = np.asarray([.5, 0., .8, -1.], np.float32)
        first.execute(action); impulse = execute_with_impulse(second, action)
        assert impulse.shape == (12, 3) and np.all(np.isfinite(impulse))
        assert np.array_equal(first.data.qpos, second.data.qpos)
        assert np.array_equal(first.data.qvel, second.data.qvel)
        assert np.array_equal(first.contact_signature(), second.contact_signature())
    finally:
        first.close(); second.close(); temporary.cleanup()


def test_contact_relabelling_is_consistent_and_invertible():
    values = np.arange(12 * 8).reshape(1, 12, 8)
    swapped = permute_contacts(values, (0, 2, 1))
    restored = permute_contacts(swapped, (0, 2, 1))
    assert np.array_equal(restored, values)
    assert np.array_equal(swapped[:, 0], values[:, 1])
    assert np.array_equal(swapped[:, 1], values[:, 0])
    assert np.array_equal(swapped[:, 4], values[:, 5])


def test_fit_only_modality_normalizers_are_finite_with_constant_channels():
    fit = np.ones((5, 24)); held = np.ones((2, 24)) * 2
    fit_z, held_z = feature_normalize(fit, held)
    assert np.all(np.isfinite(fit_z)) and np.all(np.isfinite(held_z))
    contacts = np.zeros((5, 12, 8)); means, scales = contact_normalizers(contacts)
    assert np.all(scales > 0)
    assert np.all(np.isfinite(normalize_contacts(contacts, means, scales)))
