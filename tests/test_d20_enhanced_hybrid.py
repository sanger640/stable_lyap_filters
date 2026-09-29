from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d20_enhanced_trajectory_data import contact_transition_modes, enhanced_state


def test_contact_transition_modes_cover_all_four_physical_cases():
    current = np.asarray([0, 1, 0, 1], np.float32)
    following = np.asarray([0, 1, 1, 0], np.float32)
    np.testing.assert_array_equal(contact_transition_modes(current, following), [0, 1, 2, 3])


def test_enhanced_state_has_frozen_dimension():
    class Sim:
        pass

    # Shape is checked without constructing MuJoCo by replacing the imported feature functions.
    import jenga_d20_enhanced_trajectory_data as module
    old = module.step_state, module.robot_proprioception, module.instantaneous_contacts
    module.step_state = lambda sim: np.zeros(61, np.float32)
    module.robot_proprioception = lambda sim: np.zeros(24, np.float32)
    module.instantaneous_contacts = lambda sim: np.zeros((12, 5), np.float32)
    try:
        assert enhanced_state(Sim(), np.zeros((12, 3), np.float32)).shape == (181,)
    finally:
        module.step_state, module.robot_proprioception, module.instantaneous_contacts = old
