import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_trajectory_level_ablation.py"
SPEC = importlib.util.spec_from_file_location("jenga_trajectory_level_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def test_simulation_sample_indices_are_the_requested_hold_steps():
    assert ablation.SAMPLE_INDICES == (12, 17, 27, 36, 37)


def test_dense_cache_round_trip_and_provenance(tmp_path):
    path = tmp_path / "cache.npz"
    traces = {("2", 17): np.arange(24, dtype=np.float32).reshape(2, 3, 4)}
    metadata = {"protocol_sha256": "abc", "base_sha256": "def"}

    ablation.save_cache(path, traces, metadata)
    loaded = ablation.load_cache(path, metadata)

    assert loaded.keys() == traces.keys()
    np.testing.assert_array_equal(loaded[("2", 17)], traces[("2", 17)])


def test_non_candidate_is_quiet_in_all_four_arms():
    row = {"initial_alarm": False}

    result = ablation.evaluate_row(row, None, None, np.empty((0, 8, 3)))

    assert len(result) == 4
    assert all(not value["alarm"] and not value["boundary_refined_alarm"]
               for value in result.values())
