import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_multipeak_oracle import (original_physical_responses,
                                    quaternion_rotation6)


def test_quaternion_rotation6_identity():
    result = quaternion_rotation6(np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(result, [1, 0, 0, 1, 0, 0])


def test_original_physical_response_shape():
    path = Path(__file__).resolve().parent.parent / "results/jenga/local_delta_probe_arrays.npz"
    if path.exists():
        assert original_physical_responses(path).shape == (100, 50, 27)
