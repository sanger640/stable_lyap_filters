import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_neighbor_validation import (paired_audit, select_supported,
                                        spread_by_episode)


def test_spread_by_episode_uses_multiple_episodes():
    ids = np.array(["1", "1", "1", "2", "2", "2"])
    starts = np.array([2, 10, 18, 2, 10, 18])
    selected = spread_by_episode(range(6), 4, ids, starts)
    assert set(ids[selected]) == {"1", "2"}
    assert len(selected) == 4


def test_select_supported_rejects_unsupported_and_spreads_episodes():
    rows = [
        {"episode_id": "1", "chunk_start": 2, "candidate_rank": 0,
         "neighbor_toppled_probes": 25, "supported_boundary": True},
        {"episode_id": "1", "chunk_start": 10, "candidate_rank": 1,
         "neighbor_toppled_probes": 20, "supported_boundary": True},
        {"episode_id": "2", "chunk_start": 2, "candidate_rank": 2,
         "neighbor_toppled_probes": 24, "supported_boundary": True},
        {"episode_id": "3", "chunk_start": 2, "candidate_rank": 3,
         "neighbor_toppled_probes": 49, "supported_boundary": False},
    ]
    selected = select_supported(rows, 2)
    assert [row["episode_id"] for row in selected] == ["1", "2"]
    assert all(row["stratum"] == "supported_neighbor_boundary" for row in selected)


def test_select_supported_can_report_an_explicit_shortfall():
    rows = [{"episode_id": "1", "chunk_start": 2, "candidate_rank": 0,
             "neighbor_toppled_probes": 25, "supported_boundary": True}]
    assert len(select_supported(rows, 30, allow_shortfall=True)) == 1


def test_paired_audit_reports_jump_attenuation():
    real = [{"alarm": True, "physical_boundary": True, "adjacent_jump_ratio": 8.0,
             "jump_evidence_bic": 10.0},
            {"alarm": False, "physical_boundary": False, "adjacent_jump_ratio": 2.0,
             "jump_evidence_bic": -2.0}]
    predicted = [{"alarm": False, "physical_boundary": True, "adjacent_jump_ratio": 2.0,
                  "jump_evidence_bic": 1.0},
                 {"alarm": False, "physical_boundary": False, "adjacent_jump_ratio": 1.0,
                  "jump_evidence_bic": -1.0}]
    result = paired_audit(real, predicted)
    assert result["alarm_agreement"] == .5
    assert result["supported_boundary_predicted_to_real_jump_ratio"] == .25
