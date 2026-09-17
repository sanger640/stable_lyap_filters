import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_spatial_latents import ROI_INDICES, representation


def test_spatial_representations_keep_only_selected_patches():
    tokens = np.arange(2 * 196 * 3).reshape(2, 196, 3)
    assert representation(tokens, "full").shape == (2, 196 * 3)
    flat = representation(tokens, "block_roi_flat")
    assert flat.shape == (2, len(ROI_INDICES) * 3)
    np.testing.assert_array_equal(flat.reshape(2, len(ROI_INDICES), 3), tokens[:, ROI_INDICES])
    np.testing.assert_allclose(representation(tokens, "block_roi_pool"),
                               tokens[:, ROI_INDICES].mean(1))
