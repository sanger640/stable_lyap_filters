"""Program B G0 baseline: a plain object-level graph network simulator (GNS).

One node per object, gripper part and the table; fully connected edges. Node features: two
steps of linear and angular velocity, 6D orientation, height above the table, shape (kind
one-hot and extents), inverse mass, friction, node type and, on gripper nodes, the command.
It predicts linear and angular acceleration per step and is integrated like FIGNet*.
Width, depth and MLP layout match the FIGNet* run (latent 128, 10 message-passing rounds,
2-layer MLPs), and training uses the same noise and corpus.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fignet.graph_networks import Decoder, Encoder, build_mlp  # noqa: E402
from fignet.normalization import Normalizer  # noqa: E402

import program_b_fignet as pf  # noqa: E402

KINDS = ("box", "cylinder", "wedge", "hand", "finger", "table")
STATIC_DIM = len(KINDS) + 3 + 1 + 1 + 3                 # kind, extents, inverse mass, friction, type
NODE_DIM = 3 + 3 + 3 + 3 + 6 + 1 + STATIC_DIM + 4        # velocities, rotation, height, static, command
EDGE_DIM = 3 + 1 + 3 + 3
TABLE_TOP = 0.425
ROTATION_NOISE_PER_METRE = 1.0 / 0.05                    # position noise over a 5 cm lever arm


def static_nodes(objects):
    """(n_nodes, STATIC_DIM) and moving mask; order: objects, hand, fingers, table."""
    rows, moving = [], []
    meshes = pf.gripper_meshes()
    for obj in objects:
        mesh = pf.object_mesh(obj)
        kind = np.eye(len(KINDS))[KINDS.index(obj["kind"])]
        rows.append(np.r_[kind, mesh.extents, 1.0 / (obj["density"] * mesh.volume),
                          obj["friction"], 0, 1, 0])
        moving.append(True)
    for part in pf.GRIPPER_PARTS:
        kind = np.eye(len(KINDS))[KINDS.index("hand" if part == "hand" else "finger")]
        rows.append(np.r_[kind, meshes[part].extents, 1.0 / pf.GRIPPER_MASS[part], 1.0, 0, 0, 1])
        moving.append(True)
    rows.append(np.r_[np.eye(len(KINDS))[KINDS.index("table")], 0.4, 0.8, 0.05, 0.0, 1.0, 1, 0, 0])
    moving.append(False)
    return np.asarray(rows, np.float32), np.asarray(moving)


def rotation_vectors(rotations):
    """Per-step rotation vectors between consecutive (T, n) rotation objects."""
    return [(rotations[k + 1] * rotations[k].inv()).as_rotvec() for k in range(len(rotations) - 1)]


def graph_from_history(static, moving, positions, quats, action, closed, n_objects):
    """Node and edge features from 3 frames of (n_moving, 3) positions and xyzw quats."""
    rotations = [Rotation.from_quat(q) for q in quats]
    v1, v2 = positions[1] - positions[0], positions[2] - positions[1]
    w1, w2 = rotation_vectors(rotations)
    matrix = rotations[-1].as_matrix()
    table_row = np.zeros((1, 3))
    pos = np.concatenate([positions[-1], np.array([pf.TABLE_POSE[:3]])])
    lin = np.concatenate([np.stack([v1, v2], 1).reshape(-1, 6), np.zeros((1, 6))])
    ang = np.concatenate([np.stack([w1, w2], 1).reshape(-1, 6), np.zeros((1, 6))])
    rot6 = np.concatenate([matrix[:, :, :2].reshape(-1, 6), np.array([[1, 0, 0, 1, 0, 0]])])
    height = (pos[:, 2:3] - TABLE_TOP)
    command = np.zeros((len(pos), 4))
    hand = n_objects
    site = pf.site_position(np.r_[positions[-1][hand], quats[-1][hand]])
    command[n_objects:n_objects + 3] = np.r_[np.asarray(action[:3]) - site, float(closed)]
    x = np.concatenate([lin, ang, rot6, height, static, command], 1).astype(np.float32)
    count = len(pos)
    senders, receivers = np.where(~np.eye(count, dtype=bool))
    rel = pos[senders] - pos[receivers]
    frames = np.concatenate([matrix, np.eye(3)[None]])
    local = np.einsum("nji,nj->ni", frames[receivers], rel)
    vel = np.concatenate([v2, table_row])
    e = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True), local,
                        vel[senders] - vel[receivers]], 1).astype(np.float32)
    return x, e, np.stack([senders, receivers])


def sample(static, moving, poses, actions, closed, t, n_objects, rng=None, noise_std=pf.NOISE_STD):
    """Features for predicting frame t+1; targets are clean linear/angular accelerations."""
    clean_pos = poses[t - 2:t + 2, :, :3]
    clean_quat = poses[t - 2:t + 2, :, 3:]
    positions, quats = clean_pos[:3].copy(), clean_quat[:3].copy()
    if rng is not None and noise_std:
        positions = positions + rng.normal(0, noise_std, positions.shape)
        angle = rng.normal(0, noise_std * ROTATION_NOISE_PER_METRE, positions.shape)
        quats = np.stack([(Rotation.from_rotvec(a) * Rotation.from_quat(q)).as_quat()
                          for a, q in zip(angle, quats)])
    x, e, index = graph_from_history(static, moving, positions, quats, actions[t], closed[t],
                                     n_objects)
    rotations = [Rotation.from_quat(q) for q in clean_quat]
    w = rotation_vectors(rotations)
    target_lin = clean_pos[3] - 2 * clean_pos[2] + clean_pos[1]
    target = np.concatenate([np.c_[target_lin, w[2] - w[1]], np.zeros((1, 6))]).astype(np.float32)
    return x, e, index, target, moving


def collate(samples):
    xs, es, idx, ys, masks, offset = [], [], [], [], [], 0
    for x, e, index, y, mask in samples:
        xs.append(x); es.append(e); idx.append(index + offset); ys.append(y); masks.append(mask)
        offset += len(x)
    return {"x": torch.from_numpy(np.concatenate(xs)), "e": torch.from_numpy(np.concatenate(es)),
            "index": torch.from_numpy(np.concatenate(idx, 1)).long(),
            "y": torch.from_numpy(np.concatenate(ys)),
            "mask": torch.from_numpy(np.concatenate(masks))}


class SampleBatches(torch.utils.data.Dataset):
    """Same batch recipe as program_b_fignet.GraphBatches: 8 episodes x 8 samples."""

    def __init__(self, files, batches, seed, episodes=8, per_episode=8, noise_std=pf.NOISE_STD):
        self.files = [str(f) for f in files]
        self.batches, self.seed = batches, seed
        self.episodes, self.per_episode, self.noise_std = episodes, per_episode, noise_std

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, index])
        samples = []
        for file_index in rng.choice(len(self.files), self.episodes, replace=False):
            meta, sequences = pf.episode_sequences(self.files[file_index])
            objects = meta["scene"]["objects"]
            static, moving = static_nodes(objects)
            pool = [(s, t) for s, (poses, _, _) in enumerate(sequences)
                    for t in range(2, len(poses) - 1)]
            for pick in rng.choice(len(pool), self.per_episode, replace=False):
                s, t = pool[pick]
                poses, actions, closed = sequences[s]
                samples.append(sample(static, moving, poses, actions, closed, t, len(objects),
                                      rng, self.noise_std))
        return collate(samples)


class ObjectGNS(nn.Module):
    def __init__(self, latent=128, rounds=10, mlp_layers=2, hidden=128, device="cpu"):
        super().__init__()
        self.node_encoder = Encoder(NODE_DIM, latent, mlp_layers, hidden)
        self.edge_encoder = Encoder(EDGE_DIM, latent, mlp_layers, hidden)
        self.edge_fns = nn.ModuleList([nn.Sequential(build_mlp(3 * latent, [hidden] * mlp_layers, latent),
                                                     nn.LayerNorm(latent)) for _ in range(rounds)])
        self.node_fns = nn.ModuleList([nn.Sequential(build_mlp(2 * latent, [hidden] * mlp_layers, latent),
                                                     nn.LayerNorm(latent)) for _ in range(rounds)])
        self.decoder = Decoder(latent, 6, mlp_layers, hidden)
        self.node_norm = Normalizer(NODE_DIM, name="node", device=device)
        self.edge_norm = Normalizer(EDGE_DIM, name="edge", device=device)
        self.output_norm = Normalizer(6, name="output", device=device)
        self.accumulate = True

    def forward(self, batch):
        h = self.node_encoder(self.node_norm(batch["x"], self.accumulate))
        e = self.edge_encoder(self.edge_norm(batch["e"], self.accumulate))
        senders, receivers = batch["index"]
        for edge_fn, node_fn in zip(self.edge_fns, self.node_fns):
            e = e + edge_fn(torch.cat([e, h[senders], h[receivers]], -1))
            aggregate = torch.zeros_like(h).index_add_(0, receivers, e)
            h = h + node_fn(torch.cat([h, aggregate], -1))
        return self.decoder(h)

    def loss(self, batch):
        prediction = self(batch)
        target = self.output_norm(batch["y"], self.accumulate)
        return ((prediction - target) ** 2).sum(-1)[batch["mask"]].mean()

    def accelerations(self, batch):
        return self.output_norm.inverse(self(batch))


def rollout(model, objects, poses, actions, closed, start, steps, device):
    """Predict ``steps`` model steps from true frames start-2..start; returns (steps, n+3, 7)."""
    static, moving = static_nodes(objects)
    n_objects = len(objects)
    positions = list(poses[start - 2:start + 1, :, :3])
    quats = list(poses[start - 2:start + 1, :, 3:])
    predicted = []
    for k in range(steps):
        t = start + k
        x, e, index = graph_from_history(static, moving, np.asarray(positions[-3:]),
                                         np.asarray(quats[-3:]), actions[t], closed[t], n_objects)
        batch = {"x": torch.from_numpy(x).to(device), "e": torch.from_numpy(e).to(device),
                 "index": torch.from_numpy(index).long().to(device)}
        with torch.no_grad():
            acc = model.accelerations(batch).cpu().numpy()[:-1]
        velocity = positions[-1] - positions[-2] + acc[:, :3]
        last_rotation = Rotation.from_quat(quats[-1])
        omega = (last_rotation * Rotation.from_quat(quats[-2]).inv()).as_rotvec() + acc[:, 3:]
        positions.append(positions[-1] + velocity)
        quats.append((Rotation.from_rotvec(omega) * last_rotation).as_quat())
        predicted.append(np.concatenate([positions[-1], quats[-1]], 1))
    return np.asarray(predicted)
