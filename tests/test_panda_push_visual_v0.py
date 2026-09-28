import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

import panda_push_visual_v0 as visual


def test_fixed_projection_is_deterministic_and_has_frozen_shape():
    first = visual.fixed_projection("cpu")
    second = visual.fixed_projection("cpu")
    assert first.shape == (visual.POOL_GRID ** 2 * 384, visual.VISUAL_DIMS)
    assert torch.equal(first, second)


def test_sample_indices_match_h8_plus_declared_hold_times():
    assert visual._sample_indices() == (12, 17, 27, 36, 37)


def test_summary_compares_visual_with_physical_decisions():
    visual_rows = [
        {"cohort": "topple_fork", "initial_alarm": True,
         "boundary_refined_alarm": True, "commitment_pairs": 2,
         "persistence_pairs": 2, "alarm": True, "physical_alarm": True},
        {"cohort": "safe_centered", "initial_alarm": True,
         "boundary_refined_alarm": False, "alarm": False, "physical_alarm": False},
        {"cohort": "safe_centered", "initial_alarm": True,
         "boundary_refined_alarm": True, "commitment_pairs": 2,
         "persistence_pairs": 2, "alarm": True, "physical_alarm": False},
    ]
    physical_rows = [{"alarm": True}, {"alarm": False}, {"alarm": False}]
    result = visual.summarise(visual_rows, physical_rows)
    assert result["agreement"]["exact_final_decision_agreement"] == 2
    assert result["agreement"]["visual_only_alarms"] == 1
    assert result["agreement"]["physical_alarm_recall"] == 1.0
    assert result["by_cohort"]["safe_centered"]["final_alarms"] == 1


def test_render_manifest_matches_frozen_panel_when_present():
    if not visual.RENDER_MANIFEST.exists():
        return
    manifest = json.loads(visual.RENDER_MANIFEST.read_text())
    assert manifest["protocol"] == visual.PROTOCOL
    assert manifest["source_panel_sha256"] == visual.PROTOCOL["source_panel_sha256"]
    assert len(manifest["files_sha256"]) == 56
