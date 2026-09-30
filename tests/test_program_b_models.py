from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fignet")
import program_b_fignet as pf  # noqa: E402
import program_b_gns as pg  # noqa: E402
from fignet.types import EdgeType, KinematicType, NodeType  # noqa: E402

VALIDATION = sorted((ROOT / "results/program_b/corpus/validation").glob("ep*.npz"))


def test_closed_states_follow_commands():
    actions = np.zeros((5, 4)); actions[1, 3] = 1.0; actions[3, 3] = -1.0
    assert list(pf.closed_states(actions)) == [False, True, True, False, False]
    assert list(pf.closed_states(actions[:1], initial=True)) == [True]


def test_site_is_10cm_along_hand_z():
    pose = np.r_[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    assert np.allclose(pf.site_position(pose), [0.1, 0.2, 0.4])


def test_wxyz_to_xyzw():
    assert np.allclose(pf.wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0])


@pytest.mark.skipif(not VALIDATION, reason="generic corpus not generated")
def test_gripper_parts_get_no_self_contact_edges_and_carry_commands():
    meta, sequences = pf.episode_sequences(VALIDATION[0])
    scene, obj_ids = pf.build_scene(meta["scene"]["objects"], noise_std=0.0)
    poses, actions, closed = sequences[0]
    graph = pf.make_graph(scene, obj_ids, poses, actions, closed, 2, noise=False)
    ff = graph.edge_sets[EdgeType.FACE_FACE].index
    gripper = set()
    for part in pf.GRIPPER_PARTS:
        start, end = scene._vert_index(part)
        gripper.update(range(start, end))
    if ff.shape[1]:
        both = np.isin(ff[0, :, 0], list(gripper)) & np.isin(ff[1, :, 0], list(gripper))
        assert not both.any()
    kinds = graph.node_sets[NodeType.MESH].kinematic[:, 0]
    extra = graph.node_sets[NodeType.MESH].extra
    assert np.allclose(extra[np.arange(len(kinds)), kinds], 1.0)
    assert np.all(extra[kinds != KinematicType.ACTUATED, 3:] == 0)
    command = extra[kinds == KinematicType.ACTUATED, 3:6]
    assert np.allclose(command, command[0])


@pytest.mark.skipif(not VALIDATION, reason="generic corpus not generated")
def test_gns_sample_targets_are_clean_second_differences():
    meta, sequences = pf.episode_sequences(VALIDATION[0])
    objects = meta["scene"]["objects"]
    static, moving = pg.static_nodes(objects)
    poses, actions, closed = sequences[0]
    x, e, index, y, mask = pg.sample(static, moving, poses, actions, closed, 10, len(objects),
                                     rng=np.random.default_rng(0))
    p = poses[:, :, :3]
    assert np.allclose(y[:-1, :3], p[11] - 2 * p[10] + p[9], atol=1e-6)
    assert x.shape == (len(objects) + 4, pg.NODE_DIM) and e.shape[1] == pg.EDGE_DIM
    assert mask.sum() == len(objects) + 3 and not mask[-1]
    batch = pg.collate([(x, e, index, y, mask)] * 2)
    assert batch["index"].max().item() == 2 * len(x) - 1


@pytest.mark.skipif(not VALIDATION, reason="generic corpus not generated")
def test_gns_rollout_with_zero_acceleration_extrapolates_constant_velocity():
    class Zero(torch.nn.Module):
        def accelerations(self, batch):
            return torch.zeros(len(batch["x"]), 6)
    meta, sequences = pf.episode_sequences(VALIDATION[0])
    poses, actions, closed = sequences[0]
    out = pg.rollout(Zero(), meta["scene"]["objects"], poses, actions, closed, 10, 3, "cpu")
    velocity = poses[10, :, :3] - poses[9, :, :3]
    assert np.allclose(out[2, :, :3], poses[10, :, :3] + 3 * velocity, atol=1e-9)


@pytest.mark.skipif(not VALIDATION, reason="generic corpus not generated")
def test_building_scenes_never_moves_the_shared_gripper_meshes():
    before = {k: m.vertices.copy() for k, m in pf.gripper_meshes().items()}
    for path in VALIDATION[:2]:
        meta, sequences = pf.episode_sequences(path)
        scene, obj_ids = pf.build_scene(meta["scene"]["objects"], noise_std=0.0)
        pf.make_graph(scene, obj_ids, *sequences[0], 5, noise=False)
    for key, mesh in pf.gripper_meshes().items():
        assert np.array_equal(mesh.vertices, before[key]), key


@pytest.mark.skipif(not VALIDATION, reason="generic corpus not generated")
def test_graph_vertices_stay_near_the_table():
    from fignet.types import NodeType
    for path in VALIDATION[:3]:
        meta, sequences = pf.episode_sequences(path)
        scene, obj_ids = pf.build_scene(meta["scene"]["objects"], noise_std=0.0)
        poses, actions, closed = sequences[0]
        for t in (2, 30, 60, 90):
            graph = pf.make_graph(scene, obj_ids, poses, actions, closed, t, noise=False)
            assert np.abs(graph.node_sets[NodeType.MESH].position).max() < 2.0
