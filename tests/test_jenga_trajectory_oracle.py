import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_trajectory_oracle import temporal_log_mean_evidence


def test_temporal_evidence_zero_for_no_expansion():
    assert temporal_log_mean_evidence([0, 0, 0]) == 0.0


def test_temporal_evidence_rewards_persistent_support():
    one_time = temporal_log_mean_evidence([12, -12, -12, -12])
    persistent = temporal_log_mean_evidence([12, 12, 12, -12])
    assert persistent > one_time > 0


def test_temporal_evidence_penalizes_time_search():
    value = temporal_log_mean_evidence([4, -100, -100, -100])
    np.testing.assert_allclose(value, 4 - 2 * np.log(4), rtol=1e-6)
