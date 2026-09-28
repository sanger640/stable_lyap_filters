from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

import panda_push_visual_pca as visual


def test_pca45_is_label_free_and_has_expected_shape():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(64, 5, 80)).astype(np.float32)
    model, projected = visual.fit_pca(raw)
    assert projected.shape == (64, 5, visual.PCA_DIMS)
    assert model.components_.shape == (visual.PCA_DIMS, 80)
    assert np.isfinite(projected).all()


def test_protocol_explicitly_excludes_labels_and_reference_states():
    assert visual.PROTOCOL["labels_or_reference_states_in_adapter"] is False
    assert visual.PROTOCOL["task_labels_in_monitor"] is False
