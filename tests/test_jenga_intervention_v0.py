import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_intervention_v0", ROOT / "eval/jenga_intervention_v0.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_protocol_is_fresh_fixed_and_label_blind():
    assert MODULE.EPISODES == 100
    assert MODULE.RESET_BASE == 2000
    assert MODULE.PROTOCOL["labels_in_policy"] is False
    assert MODULE.PROTOCOL["persistent_alarm_candidates"] == [1.0, .75, .5, .25, 0.0]


def test_episode_outcome_separates_pick_failure_and_safe_completion():
    positions = np.zeros((3, 3, 3))
    positions[1:, 0, 2] = .03
    positions[1:, 0, 0] = .03
    tilts = np.zeros((3, 3)); tilts[-1, 1] = 50
    trace = {"positions": list(positions), "tilts": list(tilts),
             "original_indices": [9, 10, 11]}
    result = MODULE.episode_outcome(trace, np.zeros((3, 3)))
    assert result["pick_success"]
    assert result["neighbor_failure"]
    assert not result["safe_completion"]


def test_canonical_run_has_no_force_or_episode_limit():
    source = (ROOT / "eval/jenga_intervention_v0.py").read_text()
    assert "--force" not in source
    assert "--episodes" not in source


def test_monitor_cache_key_requires_exact_state_and_action():
    snapshot = (np.arange(3.), np.arange(2.), np.arange(3.), 1.0)
    chunk = np.zeros((8, 4), np.float32)
    original = MODULE._monitor_cache_key(snapshot, chunk)
    changed = chunk.copy(); changed[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    assert MODULE._monitor_cache_key(snapshot, changed) != original
