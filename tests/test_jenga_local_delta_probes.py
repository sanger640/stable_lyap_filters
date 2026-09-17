import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_local_delta_probes import (coherent_probe_chunks, probe_scalars,
                                      read_probe_cache, read_probe_cache_with_diagnostics,
                                      remove_smooth_probe_response, write_probe_cache)


def test_probe_scalars_are_symmetric_gaussian_quantiles():
    z = probe_scalars(50)
    assert len(z) == 50
    np.testing.assert_allclose(z, -z[::-1], atol=1e-6)


def test_coherent_probes_preserve_first_pose_and_gripper():
    chunk = np.array([[1, 2, 3, -1], [2, 4, 6, 1]], np.float32)
    probes = coherent_probe_chunks(chunk, [-1, 0, 1], 0.1)
    np.testing.assert_allclose(probes[:, 0, :3], np.repeat(chunk[None, 0, :3], 3, axis=0))
    np.testing.assert_array_equal(probes[:, :, 3], np.repeat(chunk[None, :, 3], 3, axis=0))


def test_smooth_quadratic_probe_response_is_removed():
    z = probe_scalars(50)
    delta = np.stack([2 + 3 * z - z ** 2, -1 + z + 4 * z ** 2], axis=1)
    assert np.abs(remove_smooth_probe_response(delta, z)).max() < 1e-4


def test_raw_probe_cache_round_trip(tmp_path):
    rows = [{"episode_id": "3", "chunk_start": 10, "stratum": "quiet_control",
             "screen_start_tilt": 1.0, "screen_peak_tail5": 2.0,
             "screen_new_topple_tail5": False, "start_tilt": 1.5}]
    scalars = np.array([-1, 1], np.float32)
    predicted = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    actual = predicted + 1
    physical = np.array([[False, True]])
    path = tmp_path / "probes.npz"
    write_probe_cache(path, rows, scalars, predicted, actual, physical)
    metadata2, scalars2, predicted2, actual2, physical2 = read_probe_cache(path)
    assert metadata2 == rows
    np.testing.assert_array_equal(scalars2, scalars)
    np.testing.assert_array_equal(predicted2, predicted)
    np.testing.assert_array_equal(actual2, actual)
    np.testing.assert_array_equal(physical2, physical)


def test_probe_cache_preserves_physical_diagnostics(tmp_path):
    rows = [{"episode_id": "3", "chunk_start": 10, "stratum": "quiet_control",
             "screen_start_tilt": 1.0, "screen_peak_tail5": 2.0,
             "screen_new_topple_tail5": False, "start_tilt": 1.5}]
    path = tmp_path / "probes.npz"
    diagnostics = {"peak_tilt": np.ones((1, 2, 3))}
    write_probe_cache(path, rows, [-1, 1], np.zeros((1, 2, 3)),
                      np.zeros((1, 2, 3)), np.zeros((1, 2), bool), diagnostics)
    *_, loaded = read_probe_cache_with_diagnostics(path)
    np.testing.assert_array_equal(loaded["peak_tilt"], diagnostics["peak_tilt"])
