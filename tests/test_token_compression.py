import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")


def test_channel_pca_matches_numpy():
    import token_compression_test as tc
    rng = np.random.default_rng(0)
    grids = (rng.normal(size=(40, 4, 4, 12)) * np.linspace(3, 0.1, 12)).astype(np.float16)
    z = tc.ChannelPCA(3).fit(grids).transform(grids)
    flat = grids.reshape(-1, 12).astype(np.float64)
    _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    ref = (flat - flat.mean(0)) @ vt[:3].T
    assert np.allclose(np.abs(z.reshape(-1, 3)), np.abs(ref), atol=2e-3)


def test_global_pca_matches_numpy():
    import token_compression_test as tc
    rng = np.random.default_rng(1)
    grids = (rng.normal(size=(30, 3, 3, 8)) * rng.uniform(0.1, 3, size=(3, 3, 8))).astype(np.float16)
    z = tc.GlobalPCA(5).fit(grids).transform(grids)
    flat = grids.reshape(30, -1).astype(np.float64)
    _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    ref = (flat - flat.mean(0)) @ vt[:5].T
    assert np.allclose(np.abs(z), np.abs(ref), atol=2e-3)


def test_align_slots_undoes_permutation():
    import token_compression_test as tc
    rng = np.random.default_rng(2)
    base = rng.normal(size=(4, 6))
    z = np.stack([base[rng.permutation(4)] + 0.01 * rng.normal(size=(4, 6)) for _ in range(5)])
    z[0] = base
    aligned = tc.align_slots(z)
    assert np.allclose(aligned, base, atol=0.05)


@pytest.mark.parametrize("kind,size,expect", [("autoencoder", 4, (6, 7, 7, 4)), ("slots", 3, (6, 3, 256))])
def test_neural_compressor_shapes(kind, size, expect, monkeypatch):
    import token_compression_test as tc
    monkeypatch.setattr(tc, "FIT_EPOCHS", 1)
    grids = np.random.default_rng(3).normal(size=(6, 14, 14, 16)).astype(np.float16)
    assert tc.make(kind, size).fit(grids).transform(grids).shape == expect
