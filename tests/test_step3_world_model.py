import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")


def _model(kind):
    import step3_world_model as wm
    seq = torch.randn(3, 2, 20, 28, 28, 8).half()
    pairs = torch.randn(3, 2, 19, 8)
    return wm, wm.WorldModel(kind, wm.Stats(seq, pairs), 8)


@pytest.mark.parametrize("kind", ["stochastic", "deterministic"])
def test_loss_and_step(kind):
    wm, m = _model(kind)
    z = torch.randn(5, 28, 28, 8, device="cuda")
    a = torch.randn(5, 8, device="cuda")
    loss = m.loss(z, z + 0.1, a)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.step(z, a).shape == z.shape


def test_rollouts_branch_only_when_stochastic():
    wm, sto = _model("stochastic")
    _, det = _model("deterministic")
    z0 = np.zeros((28, 28, 8), np.float32)
    acts = np.zeros((4, 38, 4), np.float32)
    e10, e30 = wm.wm_rollouts(sto, z0, acts, seed=1)
    assert e10.shape == e30.shape == (4, 6272)
    assert np.std(e30, 0).max() > 0                      # noise makes identical inputs differ
    d10, d30 = wm.wm_rollouts(det, z0, acts, seed=1)
    assert np.allclose(d30, d30[0])                      # deterministic: identical inputs, identical outputs


def test_panel_actions_layout():
    import step3_world_model as wm
    chunk = np.arange(32, dtype=np.float32).reshape(8, 4)
    acts = wm.panel_actions(chunk, np.ones((2, 3), np.float32))
    assert acts.shape == (2, 38, 4)
    assert np.allclose(acts[0, :8, :3], chunk[:, :3] - 1) and np.allclose(acts[0, 8:], chunk[-1])
