from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

import panda_push_slot_data as data


def test_slot_training_specs_are_deterministic_and_locally_paired():
    first = data.sample_specs(seed=7, pairs=6)
    second = data.sample_specs(seed=7, pairs=6)
    assert first == second
    for row in first:
        offset = np.asarray(row["local_offset"])
        assert offset.shape == (data.ACTION_STEPS, 3)
        assert np.max(np.linalg.norm(offset, axis=1)) <= .0030001


def test_protocol_excludes_task_and_physical_labels():
    excluded = " ".join(data.PROTOCOL["excluded"])
    assert "topple" in excluded and "success" in excluded
    assert data.PROTOCOL["stored_fields"] == ["frames", "actions", "split", "pair_index"]
