import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from action_uncertainty import fit_tracking_error, tracking_arrays  # noqa: E402


def test_tracking_alignment_is_next_observation():
    actions = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
                        [3, 0, 0, 0]], dtype=float)
    proprio = actions.copy()
    proprio[2, 0] = .9
    proprio[3, 0] = 1.9
    design, error = tracking_arrays([SimpleNamespace(actions=actions, proprio=proprio)])
    assert design.shape == (2, 4)
    assert np.allclose(error[:, 0], [.1, .1])


def test_fit_returns_finite_covariance_and_axes():
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(5):
        action = np.cumsum(rng.normal(scale=.01, size=(30, 3)), axis=0)
        proprio = action + rng.normal(scale=.001, size=(30, 3))
        episodes.append(SimpleNamespace(actions=np.column_stack([action, np.zeros(30)]),
                                        proprio=np.column_stack([proprio, np.zeros(30)])))
    model = fit_tracking_error(episodes)
    assert model.axes.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(model.covariance) > 0)
    assert model.radial_quantile_90 > 0
