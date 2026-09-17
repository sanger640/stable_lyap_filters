import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_visual_trajectories import select_rows


def test_select_rows_has_disjoint_strata(tmp_path):
    path = tmp_path / "source.npz"
    import json
    metadata = [{"episode_id": "0", "chunk_start": 2, "stratum": "quiet_control"},
                {"episode_id": "1", "chunk_start": 2,
                 "stratum": "recentered_neighbor_boundary"}]
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)),
                        scalars=np.arange(50))
    rows, scalars = select_rows(path, 1, 1)
    assert len(rows) == 2
    assert len(scalars) == 50
