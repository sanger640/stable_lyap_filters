"""Program B: corpus episodes -> FIGNet* graphs, with an actuated Panda gripper.

Objects are free (DYNAMIC) convex meshes, the table is STATIC, and the hand and two fingers are
ACTUATED: predicted like free objects, but their nodes also carry the command (target minus the
current attachment-site position, and the open/closed state). The three gripper parts share one
group, so FIGNet builds no face-face edges between them.

Poses use the port's convention: position plus an xyzw quaternion. The corpus stores wxyz.
"""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from fignet.data_loader import ToTensor, collate_fn  # noqa: E402
from fignet.scene import Scene  # noqa: E402
from fignet.types import KinematicType  # noqa: E402
from fignet.utils import pose_to_transform, transform_to_pose  # noqa: E402
import trimesh  # noqa: E402

import program_b_corpus as corpus  # noqa: E402

GRIPPER_PARTS = ("hand", "left_finger", "right_finger")
GRIPPER_MASS = {"hand": 0.73, "left_finger": 0.015, "right_finger": 0.015}
SITE_OFFSET = np.array([0.0, 0.0, 0.1])          # attachment_site in the hand frame
FRAME_STRIDE = corpus.FRAMES_PER_STEP             # 20 ms frames per 100 ms model step
INPUT_FRAMES = 3
EXTRA_DIM = 7                                     # type one-hot (3), command (3), closed (1)
CONNECTIVITY_RADIUS = 0.005
NOISE_STD = 1.5e-4
TABLE_POSE = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])


def wxyz_to_xyzw(quat):
    quat = np.asarray(quat)
    return np.concatenate([quat[..., 1:], quat[..., :1]], -1)


def poses_xyzw(pose_wxyz):
    return np.concatenate([pose_wxyz[..., :3], wxyz_to_xyzw(pose_wxyz[..., 3:])], -1)


@lru_cache(maxsize=1)
def gripper_meshes():
    """Convex shells of each gripper body's collision geoms, in the body frame."""
    import mujoco
    import tempfile
    from jenga_short_held_tails import extract_sim
    directory = extract_sim(ROOT / "vendor/panda_express_sim.tar", tempfile.mkdtemp()).parent
    model = mujoco.MjModel.from_xml_path(str(directory / "panda.xml"))
    meshes = {}
    for body in GRIPPER_PARTS:
        points = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != model.body(body).id:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            rotation = corpus.quat_matrix(model.geom_quat[g])
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[g]
                start = model.mesh_vertadr[mesh_id]
                local = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
            else:
                h = model.geom_size[g]
                local = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            points.append(local @ rotation.T + model.geom_pos[g])
        meshes[body] = trimesh.convex.convex_hull(np.concatenate(points))
    return meshes


def object_mesh(obj):
    return trimesh.convex.convex_hull(corpus.object_vertices(obj))


def properties(inverse_mass, friction):
    return {"mass": float(inverse_mass), "friction": [float(friction), 0.005, 0.0001],
            "restitution": 0.0}


def build_scene(objects, noise_std=NOISE_STD):
    """Port Scene for one episode; returns (scene, obj_ids)."""
    scene = Scene({"env": {}, "objects": {}, "connectivity_radius": CONNECTIVITY_RADIUS,
                   "noise_std": noise_std})
    table = trimesh.creation.box(extents=[0.4, 0.8, 0.05])
    table.apply_transform(pose_to_transform(TABLE_POSE))
    scene.add_object("table", KinematicType.STATIC, table)
    scene._init_poses["table"] = TABLE_POSE
    scene._set_obj_properties("table", properties(0.0, 1.0))
    obj_ids = {}
    for number, obj in enumerate(objects):
        mesh = object_mesh(obj)
        mass = obj["density"] * mesh.volume
        name = f"obj{number}"
        scene.add_object(name, KinematicType.DYNAMIC, mesh)
        scene._set_obj_properties(name, properties(1.0 / mass, obj["friction"]))
        obj_ids[name] = number
    for offset, part in enumerate(GRIPPER_PARTS):
        scene.add_object(part, KinematicType.ACTUATED, gripper_meshes()[part])
        scene._set_obj_properties(part, properties(1.0 / GRIPPER_MASS[part], 1.0))
        scene.set_group(part, "gripper")
        obj_ids[part] = len(objects) + offset
    scene._dim_properties = scene._object_property_array("table").shape[1]
    scene._setup()
    return scene, obj_ids


def site_position(hand_pose_xyzw):
    transform = pose_to_transform(hand_pose_xyzw)
    return transform[:3, :3] @ SITE_OFFSET + transform[:3, 3]


def closed_states(actions, initial=False):
    """Gripper state during each step: closed after a close command until an open command."""
    state, states = bool(initial), []
    for command in actions[:, 3]:
        if command > 0.9:
            state = True
        elif command < -0.9:
            state = False
        states.append(state)
    return np.asarray(states)


def node_extra(scene, obj_ids, hand_pose_xyzw, action, closed):
    """(mesh_extra, obj_extra): type one-hot for every node; command and state on gripper nodes."""
    kin_mesh, kin_obj = scene._node_types()
    command = np.r_[np.asarray(action[:3]) - site_position(hand_pose_xyzw), float(closed)]

    def fill(kinds):
        extra = np.zeros((len(kinds), EXTRA_DIM))
        extra[np.arange(len(kinds)), kinds[:, 0]] = 1.0
        actuated = kinds[:, 0] == KinematicType.ACTUATED
        extra[actuated, 3:] = command
        return extra
    return fill(kin_mesh), fill(kin_obj)


def episode_sequences(path):
    """Model-rate sequences of one episode: [(poses (T, n+3, 7) xyzw, actions (T-1, 4), closed)]."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))

    def poses(objects, robot):
        return poses_xyzw(np.concatenate([objects, robot], 1)[::FRAME_STRIDE])

    actions = data["actions"]
    closed = closed_states(actions)
    sequences = [(poses(data["main_objects"], data["main_robot"]), actions, closed)]
    branch_step = int(data["branch_step"])
    if branch_step >= 0:
        start_state = bool(closed[branch_step - 1]) if branch_step > 0 else False
        for j in range(len(data["branch_actions"])):
            branch_actions = data["branch_actions"][j]
            sequences.append((poses(data["branch_objects"][j], data["branch_robot"][j]),
                              branch_actions, closed_states(branch_actions, start_state)))
    return meta, sequences


def make_graph(scene, obj_ids, poses, actions, closed, t, noise=True):
    """Graph for predicting frame t+1 from frames t-2..t under action t."""
    scene.synchronize_states(poses[t - INPUT_FRAMES + 1:t + 1], obj_ids)
    hand = poses[t, obj_ids["hand"]]
    scene.set_extra(*node_extra(scene, obj_ids, hand, actions[t], closed[t]))
    return scene.to_graph(target_poses=poses[t + 1], obj_ids=obj_ids, noise=noise)


class GraphBatches(torch.utils.data.Dataset):
    """Each item is one collated batch: ``episodes`` random episodes x ``per_episode`` samples.

    Grouping by episode builds each Scene (meshes, collision BVHs) once per batch instead of once
    per graph. Item i is deterministic given ``seed``.
    """

    def __init__(self, files, batches, seed, episodes=8, per_episode=8, noise_std=NOISE_STD):
        self.files = [str(f) for f in files]
        self.batches, self.seed = batches, seed
        self.episodes, self.per_episode, self.noise_std = episodes, per_episode, noise_std

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, index])
        graphs = []
        for file_index in rng.choice(len(self.files), self.episodes, replace=False):
            meta, sequences = episode_sequences(self.files[file_index])
            scene, obj_ids = build_scene(meta["scene"]["objects"], self.noise_std)
            samples = [(s, t) for s, (poses, _, _) in enumerate(sequences)
                       for t in range(INPUT_FRAMES - 1, len(poses) - 1)]
            for pick in rng.choice(len(samples), self.per_episode, replace=False):
                s, t = samples[pick]
                poses, actions, closed = sequences[s]
                graphs.append(ToTensor()(make_graph(scene, obj_ids, poses, actions, closed, t)))
        return collate_fn(graphs)


def rollout(simulator, scene, obj_ids, poses, actions, closed, start, steps, device):
    """Predict ``steps`` model steps from true frames start-2..start; returns (steps, n+3, 7)."""
    current = poses[start - INPUT_FRAMES + 1:start + 1].copy()
    scene.synchronize_states(current, obj_ids)
    last = current[-1].copy()
    predicted = []
    for k in range(steps):
        t = start + k
        scene.set_extra(*node_extra(scene, obj_ids, last[obj_ids["hand"]], actions[t], closed[t]))
        graph = ToTensor(device)(scene.to_graph())
        with torch.no_grad():
            m_acc, o_acc = simulator.predict_accelerations(graph)
            m_acc = simulator.denormalize_accelerations(m_acc)
            o_acc = simulator.denormalize_accelerations(o_acc)
        relative = scene.update(m_acc=m_acc.cpu().numpy(), o_acc=o_acc.cpu().numpy(),
                                obj_ids=obj_ids, device=device)
        for name, index in obj_ids.items():
            last[index] = transform_to_pose(pose_to_transform(relative[index])
                                            @ pose_to_transform(last[index]))
        predicted.append(last.copy())
    return np.asarray(predicted)
