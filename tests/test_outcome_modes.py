import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from outcome_modes import corner_displacements_mm, same_partition, two_mode_test  # noqa: E402

HALF = (0.0125, 0.01, 0.0375)


def identity_pose():
    pos = np.zeros(9)
    rot = np.tile(np.eye(3)[:, :2].reshape(-1), 3)
    return np.concatenate([pos, rot])


def test_pure_translation_moves_every_corner_equally():
    start = identity_pose()
    delta = np.zeros(27); delta[3] = 0.002  # block 1 moves 2 mm in x
    d = corner_displacements_mm(start, delta, HALF, [1]).reshape(8, 3)
    assert np.allclose(d, [2, 0, 0])


def test_rotation_moves_corners():
    start = identity_pose()
    end = start.copy()
    tilt = np.array([[1, 0], [0, 0], [0, 1]], float)  # 90 deg about x: y->z, z->-y
    end[9 + 6:9 + 12] = tilt.reshape(-1)
    d = corner_displacements_mm(start, end - start, HALF, [1]).reshape(8, 3)
    assert np.abs(d).max() > 40


def test_two_separated_modes_alarm_single_spread_does_not():
    rng = np.random.default_rng(0)
    one = rng.normal(0, 1, (64, 6))
    assert not two_mode_test(one, perturbation_mm=1.0).alarm
    two = one.copy(); two[:10] += 40
    result = two_mode_test(two, perturbation_mm=1.0)
    assert result.alarm and result.minority == 10


def test_tiny_separated_modes_fail_the_consequence_floor():
    rng = np.random.default_rng(1)
    x = rng.normal(0, .01, (64, 6)); x[:20] += .5
    result = two_mode_test(x, perturbation_mm=3.0)
    assert result.ashman_d > 2 and not result.alarm


def test_same_partition_ignores_label_swap():
    a = np.array([0, 0, 1, 1, 1])
    assert same_partition(a, 1 - a)
    assert not same_partition(a, np.array([1, 0, 1, 0, 1]))


from outcome_modes import groupings_persist, multi_mode_test  # noqa: E402


def test_multi_single_blob_is_one_group():
    x = np.random.default_rng(0).normal(0, 1, (64, 20))
    assert multi_mode_test(x)["groups"] == 1


def test_multi_elongated_continuous_spread_is_one_group():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (64, 20)); x[:, 0] *= 10
    assert multi_mode_test(x)["groups"] == 1


def test_multi_finds_small_group_off_the_main_direction():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (64, 20)); x[:, 0] *= 10  # arm-like dominant spread
    x[:4, 1] += 30  # 4 runs fork along a different direction
    result = multi_mode_test(x)
    assert result["groups"] == 2 and result["sizes"][-1] == 4


def test_multi_three_groups():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (64, 20)); x[:10, 0] += 25; x[10:18, 1] += 25
    assert multi_mode_test(x)["groups"] == 3


def test_groupings_persist_allows_merge_not_reshuffle():
    a = np.array([0] * 10 + [1] * 5 + [2] * 5)
    assert groupings_persist(a, np.array([0] * 10 + [1] * 10))
    assert not groupings_persist(a, np.array([0, 1] * 10))


from outcome_modes import coarse_persistent_fork  # noqa: E402


def test_coarse_persistence_accepts_reshuffled_subgroups():
    # topple (runs 0-19) vs stable (20-63); topple subgroups reshuffle between times
    a = np.array([0] * 10 + [1] * 10 + [2] * 44)
    b = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [2] * 44)
    assert coarse_persistent_fork(a, b)
    assert not groupings_persist(a, b)


def test_coarse_persistence_rejects_random_cuts_and_single_group():
    rng = np.random.default_rng(0)
    rejected = sum(not coarse_persistent_fork(rng.integers(0, 2, 64), rng.integers(0, 2, 64))
                   for _ in range(50))
    assert rejected == 50
    assert not coarse_persistent_fork(np.zeros(64, int), np.zeros(64, int))


def test_coarse_persistence_stray_tolerance():
    a = np.array([0] * 60 + [1] * 4)
    b = a.copy(); b[0] = 1  # one run switches
    assert coarse_persistent_fork(a, b)
    b[1] = 1; b[61] = 0  # now 3 strays: pairs (0,1)x2 is linked, (1,0)x1 stray -> merges
    assert not coarse_persistent_fork(a, b)


from outcome_modes import dominant_binary  # noqa: E402


def test_dominant_split_takes_the_widest_gap():
    z = np.array([[0.], [0.2], [5.], [5.2], [0.1], [5.1]])
    labels = np.array([0, 1, 2, 3, 0, 2])  # four groups, the wide gap is 0/1 vs 2/3
    d = dominant_binary(z, labels)
    assert set(d[labels <= 1]) == {d[0]} and set(d[labels >= 2]) == {d[2]} and d[0] != d[2]


def test_dominant_split_of_toppled_subgroups_is_topple_vs_stable():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (64, 5))
    z[:6, 1] += 40; z[6:12, 1] += 40; z[6:12, 2] += 8  # two topple sub-poses
    result = multi_mode_test(z)
    assert result["groups"] >= 2
    dom = result["dominant"]
    assert set(dom[:12]) == {dom[0]} and dom[0] != dom[20]
