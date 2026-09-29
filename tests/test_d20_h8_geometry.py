from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from enhanced_hybrid_dynamics import EnhancedContactModeGNN
from jenga_d20_h8_geometry import correlation, model_inputs, model_scales, unroll_enhanced


def test_correlation_handles_linear_and_degenerate_inputs():
    assert np.isclose(correlation(np.arange(5), np.arange(5) * 2), 1.)
    assert correlation(np.ones(5), np.arange(5)) == 0.


def test_enhanced_unroll_shape():
    model = EnhancedContactModeGNN(hidden=32, rounds=1)
    norms = {"state_mean": np.zeros(181, np.float32), "state_scale": np.ones(181, np.float32),
             "action_mean": np.zeros(4, np.float32), "action_scale": np.ones(4, np.float32),
             "block_scale": np.ones(15, np.float32), "grip_scale": np.ones(3, np.float32),
             "proprio_scale": np.ones(24, np.float32), "force_mean": np.zeros(96, np.float32),
             "force_scale": np.ones(96, np.float32)}
    state = torch.zeros(3, 181); state[:, 9:27] = torch.tensor([1., 0., 0., 1., 0., 0.] * 3)
    values, outputs = unroll_enhanced(model, state, torch.zeros(3, 2, 4),
                                      model_inputs(norms, "cpu"), model_scales(norms, "cpu"))
    assert values.shape == (3, 2, 181)
    assert len(outputs) == 2
