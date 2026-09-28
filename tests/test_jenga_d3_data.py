import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("jenga_d3_data", ROOT / "eval/jenga_d3_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_action_sequence_uses_common_nominal_hold():
    nominal = np.zeros((8, 4), np.float32)
    nominal[:, :3] = np.arange(8)[:, None]
    nominal[:, 3] = -1
    residual = np.ones((8, 3), np.float32)
    sequence = MODULE.action_sequence(nominal, residual, 2.0)
    assert sequence.shape == (38, 4)
    assert np.allclose(sequence[:8, :3], nominal[:, :3] + 2)
    assert np.array_equal(sequence[8:], np.repeat(nominal[-1:], 30, axis=0))


def test_refinement_halves_width_and_preserves_step_boundary():
    def simulate(alpha):
        trace = np.zeros((38, 61), np.float32)
        trace[:, 0] = -1 if alpha < 0 else 1
        return trace

    left, right = simulate(-.5), simulate(.5)
    scales = MODULE.generic_scales(np.stack([left, right]))
    alphas, traces, sides = MODULE.refine_line(-.5, .5, left, right, simulate, scales)
    widths = alphas[:, 1] - alphas[:, 0]
    assert np.allclose(widths, 2.0 ** -np.arange(MODULE.LEVELS))
    assert np.all(traces[:, 0, :, 0] == -1)
    assert np.all(traces[:, 1, :, 0] == 1)
    assert len(sides) == MODULE.LEVELS - 1


def test_refinement_records_smooth_collapse_too():
    def simulate(alpha):
        trace = np.zeros((38, 61), np.float32); trace[:, 0] = alpha
        return trace

    coarse = np.stack([simulate(alpha) for alpha in (-.5, 0, .5)])
    scales = MODULE.generic_scales(coarse)
    alphas, traces, _ = MODULE.refine_line(-.5, .5, coarse[0], coarse[2], simulate, scales)
    gaps = np.abs(traces[:, 1, -1, 0] - traces[:, 0, -1, 0])
    assert np.allclose(gaps, alphas[:, 1] - alphas[:, 0])
