import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_all_block_tails import analyse


def test_tail_analysis_counts_middle_and_neighbor_failures():
    arrays = {"start_tilt": np.zeros((2, 3)), "start_position": np.zeros((2, 3, 3))}
    for tail in (0, 5, 10):
        peak = np.array([[50, 0, 0], [0, 50, 0]], float)
        arrays[f"tail{tail}_peak_tilt"] = peak
        arrays[f"tail{tail}_min_z"] = np.full((2, 3), 0.46)
        arrays[f"tail{tail}_end_position"] = np.zeros((2, 3, 3))
    result = analyse(arrays)
    assert result["tails"]["5"]["middle_failures"] == 1
    assert result["tails"]["5"]["neighbor_failures"] == 1
    assert result["tails"]["5"]["any_failures"] == 2
