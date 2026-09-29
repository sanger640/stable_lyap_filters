from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d12_coverage_data import STRATA, classify, select_balanced
from jenga_d12_verify import fold_for, source_identity
from jenga_d12_combined_manifest import validate


def row(index, **updates):
    value = {"path": f"p{index}", "source": f"ep{index}_seed100", "episode": str(index),
             "seed": 100, "state_index": 0, "probe_index": 0, "step": 3,
             "contact_create": False, "contact_loss": False,
             "velocity_change": .1, "pose_change": .1, "persistent_motion": .1}
    value.update(updates); return value


def cuts():
    return {"velocity_change_high": .8, "pose_change_high": .8,
            "persistent_motion_high": .8, "velocity_change_quiet": .2,
            "pose_change_quiet": .2, "persistent_motion_quiet": .2}


def test_classification_priority_and_quiet_definition():
    assert classify(row(0, contact_create=True, contact_loss=True), cuts()) == "contact_create"
    assert classify(row(1, contact_loss=True), cuts()) == "contact_loss"
    assert classify(row(2, velocity_change=.9), cuts()) == "velocity_impulse"
    assert classify(row(3, pose_change=.9), cuts()) == "pose_motion"
    assert classify(row(4, persistent_motion=.9), cuts()) == "persistent_motion"
    assert classify(row(5), cuts()) == "quiet"
    assert classify(row(6, velocity_change=.5), cuts()) is None


def test_balanced_selection_is_deterministic_and_path_distinct():
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        for index in range(5):
            kwargs = ({"contact_create": True} if stratum == "contact_create" else
                      {"contact_loss": True} if stratum == "contact_loss" else
                      {"velocity_change": .9} if stratum == "velocity_impulse" else
                      {"pose_change": .9} if stratum == "pose_motion" else
                      {"persistent_motion": .9} if stratum == "persistent_motion" else {})
            rows.append(row(10 * stratum_index + index, **kwargs))
    selected, available = select_balanced(rows, cuts(), 3, 4)
    again, _ = select_balanced(rows, cuts(), 3, 4)
    assert selected == again
    assert len(selected) == 18
    assert all(available[name] == 5 for name in STRATA)
    keys = {(item["source"], item["state_index"], item["probe_index"]) for item in selected}
    assert len(keys) == len(selected)


def test_frozen_two_axis_split_is_disjoint():
    protocol = {"episode_axis": {"validation_episode_ids": [1]},
                "configuration_axis": {"validation_reset_seeds": [109]}}
    assert source_identity("ep12_seed104") == (12, 104)
    assert fold_for(2, 108, protocol) == "fit"
    assert fold_for(1, 108, protocol) == "episode_validation"
    assert fold_for(2, 109, protocol) == "configuration_validation"
    assert fold_for(1, 109, protocol) == "joint_validation"


def test_combined_manifest_rejects_reserved_fit_source():
    split = {"episode_axis": {"validation_episode_ids": [1]},
             "configuration_axis": {"validation_reset_seeds": [109]}}
    valid = [{"file": "ep2_seed100.npz", "state_index": 0, "probe_index": 0,
              "contact_step": 3, "episode": 2, "reset_seed": 100, "fold": "fit"}]
    assert validate(valid, split) == []
    invalid = [{**valid[0], "episode": 1}]
    assert any("reserved source in fit" in error for error in validate(invalid, split))
