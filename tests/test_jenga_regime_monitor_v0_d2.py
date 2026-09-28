import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_regime_monitor_v0_d2", ROOT / "eval/jenga_regime_monitor_v0_d2.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_probe_windows_reconstruct_frozen_layout():
    rng = np.random.default_rng(4)
    snippets = rng.normal(size=(6, 8, 3)).astype(np.float32)
    nominal = rng.normal(size=(8, 4)).astype(np.float32)
    history = rng.normal(size=(2, 4)).astype(np.float32)
    windows = []
    for noise in snippets:
        probe = nominal.copy(); probe[:, :3] -= noise
        windows.append(np.concatenate([history, probe, np.repeat(nominal[-1:], 30, axis=0)]))
    windows = np.stack(windows)
    rebuilt = MODULE.probe_windows(windows, snippets, snippets)
    assert np.allclose(rebuilt, windows, rtol=1e-6, atol=1e-7)


def test_probe_windows_preserve_gripper_and_common_hold():
    snippets = np.zeros((2, 8, 3), np.float32)
    snippets[1] = 0.25
    windows = np.zeros((2, 40, 4), np.float32)
    windows[:, 2:10, 3] = np.arange(8)
    windows[:, 10:] = windows[0, 9]
    changed = MODULE.probe_windows(windows, snippets, np.full((1, 8, 3), 0.5, np.float32))
    assert np.array_equal(changed[0, 2:10, 3], np.arange(8))
    assert np.all(changed[0, 2:10, :3] == -0.5)
    assert np.all(changed[0, 10:] == windows[0, 9])


def test_candidate_summary_exposes_each_frozen_stage():
    rows = [
        {"class": "topple_fork", "initial_alarm": True,
         "boundary_refined_alarm": True, "commitment_pairs": 2,
         "persistence_pairs": 3, "alarm": True},
        {"class": "topple_fork", "initial_alarm": True,
         "boundary_refined_alarm": True, "commitment_pairs": 1,
         "persistence_pairs": 3, "alarm": False},
        {"class": "quiet", "initial_alarm": False,
         "boundary_refined_alarm": False, "alarm": False},
    ]
    result = MODULE.candidate_summary(rows)
    assert result["topple_fork"] == {
        "states": 2, "initial_candidates": 2, "boundary_candidates": 2,
        "commitment_majority": 1, "persistence_majority": 2, "v0_alarms": 1}
    assert result["quiet"]["v0_alarms"] == 0


def test_evaluator_does_not_import_simulator_rollouts():
    source = (ROOT / "eval/jenga_regime_monitor_v0_d2.py").read_text()
    assert "DirectJengaSim" not in source
    assert "rollout_trace" not in source
