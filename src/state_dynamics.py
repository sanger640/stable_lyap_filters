"""Step-wise dynamics on privileged object state: a small graph net integrated forward.

PLAN_WORLDMODEL W5, the external plan's Phase 3 minimum implementation run oracle-state-first.
The outcome is obtained by INTEGRATING a learned step function, never predicted in one shot, so a
discontinuity can emerge from contact rather than having to be fitted directly.

State after every action (61 numbers, the layout `eval/jenga_state_data.py --per-step` records):

    [0:9]    block positions, block b at 3b          (middle, left, right)
    [9:27]   block rotation, first two columns, block b at 9 + 6b
    [27:45]  block velocity, block b at 27 + 6b: linear xyz then angular xyz
    [45:57]  contact flags, in jenga_short_held_tails.CONTACT_LABELS order
    [57:61]  gripper position xyz, gripper closed flag

Graph: three block nodes and one gripper node, fully connected. Block nodes carry position
relative to the gripper, absolute height, orientation, velocity and their own support/robot
contacts; the gripper node carries the commanded target relative to where the gripper is. Edges
carry relative position, distance and the pair's contact flag. Message passing is shared across
blocks, so the model is equivariant to relabelling them.

Head: a Gaussian per predicted quantity (the single-expert stochastic baseline the plan asks for
first), plus contact logits for the next step.
"""
import numpy as np
import torch
import torch.nn as nn

POS, ROT, VEL, CONTACT, GRIPPER = slice(0, 9), slice(9, 27), slice(27, 45), slice(45, 57), slice(57, 61)
STATE_DIM = 61
N_BLOCKS = 3
# CONTACT_LABELS: middle-left, middle-right, left-right, then block-table x3, block-floor x3,
# block-robot x3. Block-block pairs map onto graph edges; the rest are per-block node features.
BLOCK_PAIRS = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
SUPPORT = {b: (3 + b, 6 + b) for b in range(N_BLOCKS)}   # table, floor contact indices
ROBOT = {b: 9 + b for b in range(N_BLOCKS)}
ACTION_SCALE_M = 0.0032      # measured execution-error scale; actions arrive as O(1) numbers
POSITION_SCALE_M = 0.05      # block size is 25 x 20 x 75 mm

BLOCK_IN = 3 + 1 + 6 + 6 + 3      # rel pos, height, rot6, vel, own contacts (table/floor/robot)
GRIPPER_IN = 3 + 1 + 1 + 1        # target - position, height, closed flag, close command
EDGE_IN = 3 + 1 + 1 + 2           # relative position, distance, contact, edge type
BLOCK_OUT = 3 + 6 + 6             # d position, d rotation, d velocity
GRIPPER_OUT = 3                   # d gripper position


def _mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                         nn.GELU(), nn.Linear(hidden, out))


def node_and_edge_features(state, action):
    """state (B, 61), action (B, 4) absolute target -> node (B, 4, *), edge (B, 12, *) features.

    Nodes 0-2 are blocks, node 3 is the gripper. Edges are all ordered pairs.
    """
    batch = state.shape[0]
    positions = state[:, POS].view(batch, N_BLOCKS, 3)
    gripper = state[:, GRIPPER]
    grip_pos = gripper[:, :3]
    contacts = state[:, CONTACT]

    block_nodes = []
    for b in range(N_BLOCKS):
        rel = (positions[:, b] - grip_pos) / POSITION_SCALE_M
        height = positions[:, b, 2:3] / POSITION_SCALE_M
        rot = state[:, 9 + 6 * b: 15 + 6 * b]
        vel = state[:, 27 + 6 * b: 33 + 6 * b]
        own = torch.stack([contacts[:, SUPPORT[b][0]], contacts[:, SUPPORT[b][1]],
                           contacts[:, ROBOT[b]]], dim=1)
        block_nodes.append(torch.cat([rel, height, rot, vel, own], dim=1))
    gripper_node = torch.cat([
        (action[:, :3] - grip_pos) / ACTION_SCALE_M,
        grip_pos[:, 2:3] / POSITION_SCALE_M,
        gripper[:, 3:4],
        torch.sign(action[:, 3:4]) * (action[:, 3:4].abs() > 0.9)], dim=1)

    node_pos = torch.cat([positions, grip_pos[:, None]], dim=1)          # (B, 4, 3)
    senders, receivers, edges = [], [], []
    for s in range(N_BLOCKS + 1):
        for r in range(N_BLOCKS + 1):
            if s == r:
                continue
            rel = (node_pos[:, r] - node_pos[:, s]) / POSITION_SCALE_M
            if s < N_BLOCKS and r < N_BLOCKS:
                contact = contacts[:, BLOCK_PAIRS[tuple(sorted((s, r)))]]
                kind = torch.tensor([1.0, 0.0], device=state.device).expand(batch, 2)
            else:
                block = s if s < N_BLOCKS else r
                contact = contacts[:, ROBOT[block]]
                kind = torch.tensor([0.0, 1.0], device=state.device).expand(batch, 2)
            edges.append(torch.cat([rel, rel.norm(dim=1, keepdim=True), contact[:, None], kind],
                                   dim=1))
            senders.append(s); receivers.append(r)
    return (torch.stack(block_nodes, dim=1), gripper_node, torch.stack(edges, dim=1),
            torch.tensor(senders, device=state.device), torch.tensor(receivers, device=state.device))


class StepGraphNet(nn.Module):
    """One integration step: (state, action) -> Gaussian over the state change, contact logits."""

    def __init__(self, hidden=128, rounds=3):
        super().__init__()
        self.block_encoder = _mlp(BLOCK_IN, hidden, hidden)
        self.gripper_encoder = _mlp(GRIPPER_IN, hidden, hidden)
        self.edge_encoder = _mlp(EDGE_IN, hidden, hidden)
        self.edge_updates = nn.ModuleList([_mlp(3 * hidden, hidden, hidden) for _ in range(rounds)])
        self.node_updates = nn.ModuleList([_mlp(2 * hidden, hidden, hidden) for _ in range(rounds)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.block_head = _mlp(hidden, 2 * BLOCK_OUT + 3, hidden)     # mean, log var, 3 contacts
        self.gripper_head = _mlp(hidden, 2 * GRIPPER_OUT, hidden)
        self.pair_head = _mlp(hidden, 1, hidden)                      # block-block contact logit

    def trunk(self, state, action):
        """Shared encoder and message passing -> node and edge embeddings plus raw block features."""
        blocks, grip, edges, senders, receivers = node_and_edge_features(state, action)
        nodes = torch.cat([self.block_encoder(blocks), self.gripper_encoder(grip)[:, None]], dim=1)
        e = self.edge_encoder(edges)
        for edge_up, node_up, edge_norm, node_norm in zip(
                self.edge_updates, self.node_updates, self.edge_norms, self.node_norms):
            e = edge_norm(e + edge_up(torch.cat([e, nodes[:, senders], nodes[:, receivers]], -1)))
            incoming = torch.zeros_like(nodes).index_add_(1, receivers, e)
            nodes = node_norm(nodes + node_up(torch.cat([nodes, incoming], dim=-1)))
        return nodes, e, senders, receivers, blocks

    def shared_heads(self, nodes, e, senders, receivers):
        grip_out = self.gripper_head(nodes[:, N_BLOCKS])                 # (B, 6)
        pair_logits = torch.stack([
            self.pair_head(e[:, i]).squeeze(-1) for i, (s, r) in enumerate(
                zip(senders.tolist(), receivers.tolist())) if s < r < N_BLOCKS], dim=1)
        return {"gripper_mean": grip_out[:, :GRIPPER_OUT],
                "gripper_logvar": grip_out[:, GRIPPER_OUT:].clamp(-10, 5),
                "pair_contact_logits": pair_logits}                         # (0,1), (0,2), (1,2)

    def forward(self, state, action):
        nodes, e, senders, receivers, _ = self.trunk(state, action)
        block_out = self.block_head(nodes[:, :N_BLOCKS])                 # (B, 3, 2*15 + 3)
        return {"block_mean": block_out[..., :BLOCK_OUT],
                "block_logvar": block_out[..., BLOCK_OUT:2 * BLOCK_OUT].clamp(-10, 5),
                "block_contact_logits": block_out[..., 2 * BLOCK_OUT:],     # table, floor, robot
                **self.shared_heads(nodes, e, senders, receivers)}


class StepGraphMoE(StepGraphNet):
    """The hybrid variant: K local experts per block, switched by an action-conditioned gate.

    Hybrid-system reading (PLAN_WORLDMODEL W5): each expert is smooth within one regime; the gate
    pi_k(s_t, a_t) is the guard; the per-block, per-step one-hot choice is the discrete regime
    latent. The gate reads the block's node embedding -- which carries the action through the
    gripper's messages -- plus that block's raw contact flags and distance to the gripper, the
    contact/event features that trigger switches.

    Switching is HARD: straight-through Gumbel-softmax selects exactly one expert in the forward
    pass, while gradients flow through the soft probabilities. In eval mode the argmax is used, so
    mean rollouts are deterministic. Expert indices are not physical modes without an audit.
    """

    GATE_EXTRA = 4   # the block's own table/floor/robot contacts + distance to the gripper

    def __init__(self, hidden=128, rounds=3, experts=3):
        super().__init__(hidden, rounds)
        del self.block_head
        self.experts = experts
        self.expert_heads = nn.ModuleList([_mlp(hidden, 2 * BLOCK_OUT, hidden)
                                           for _ in range(experts)])
        self.block_contact_head = _mlp(hidden, 3, hidden)
        self.gate = _mlp(hidden + self.GATE_EXTRA, experts, hidden)
        self.temperature = 1.0

    def forward(self, state, action):
        nodes, e, senders, receivers, blocks = self.trunk(state, action)
        block_nodes = nodes[:, :N_BLOCKS]                                # (B, 3, H)
        event = torch.cat([blocks[..., 16:19], blocks[..., 0:3].norm(dim=-1, keepdim=True)], -1)
        logits = self.gate(torch.cat([block_nodes, event], dim=-1))     # (B, 3, K)
        if self.training:
            uniform = torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)
            soft = ((logits - torch.log(-torch.log(uniform))) / self.temperature).softmax(-1)
        else:
            soft = (logits / self.temperature).softmax(-1)
        hard = nn.functional.one_hot(soft.argmax(-1), self.experts).to(soft.dtype)
        regime = hard - soft.detach() + soft                             # straight-through
        outputs = torch.stack([head(block_nodes) for head in self.expert_heads], dim=2)  # B,3,K,30
        chosen = (regime.unsqueeze(-1) * outputs).sum(2)                 # (B, 3, 30)
        return {"block_mean": chosen[..., :BLOCK_OUT],
                "block_logvar": chosen[..., BLOCK_OUT:].clamp(-10, 5),
                "block_contact_logits": self.block_contact_head(block_nodes),
                "gate_probabilities": logits.softmax(-1), "regime": hard,
                **self.shared_heads(nodes, e, senders, receivers)}


def load_balance(gate_probabilities):
    """Collapse regularisation: KL between the batch's mean expert usage and uniform usage."""
    usage = gate_probabilities.reshape(-1, gate_probabilities.shape[-1]).mean(0)
    return (usage * (usage.clamp_min(1e-9) * usage.shape[0]).log()).sum()


def neighbour_tilt_deg(states):
    """Tilt of blocks 1 and 2 from the predicted rotation: arccos of the body z-axis's z.

    all_block_pose stores xmat[:, :, :2] flattened ROW-major, so the six numbers per block are
    (r00, r01, r10, r11, r20, r21): column 0 is the even entries, column 1 the odd ones. Reading
    them as two contiguous columns reports every block as ~90 degrees tilted.
    """
    tilts = []
    for b in (1, 2):
        block = states[..., 9 + 6 * b: 15 + 6 * b]
        c0, c1 = block[..., 0::2], block[..., 1::2]
        c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True).clip(1e-9)
        c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True).clip(1e-9)
        z = np.cross(c0, c1)[..., 2]
        tilts.append(np.degrees(np.arccos(np.clip(z, -1, 1))))
    return np.stack(tilts, axis=-1)


def delta_targets(state, next_state):
    """True per-step change, in the model's output layout: blocks (B, 3, 15), gripper (B, 3)."""
    batch = state.shape[0]
    d = next_state - state
    blocks = torch.cat([d[:, POS].view(batch, 3, 3), d[:, ROT].view(batch, 3, 6),
                        d[:, VEL].view(batch, 3, 6)], dim=-1)
    return blocks, d[:, 57:60]


def contact_targets(next_state):
    c = next_state[:, CONTACT]
    per_block = torch.stack([torch.stack([c[:, SUPPORT[b][0]], c[:, SUPPORT[b][1]], c[:, ROBOT[b]]],
                                         dim=1) for b in range(N_BLOCKS)], dim=1)
    return per_block, c[:, [0, 1, 2]]


def apply_step(state, action, out, delta_scale, sample=False, noise=None):
    """Integrate one step: add the (normalised) predicted change, set contacts from the logits.

    delta_scale: (blocks (15,), gripper (3,)) standard deviations of the true per-step change,
    used to un-normalise. With sample=True a Gaussian draw replaces the mean; pass `noise` to use
    common random numbers across probes.
    """
    block_scale, grip_scale = delta_scale
    block_delta = out["block_mean"]
    grip_delta = out["gripper_mean"]
    if sample:
        eps_b, eps_g = noise if noise is not None else (
            torch.randn_like(block_delta), torch.randn_like(grip_delta))
        block_delta = block_delta + eps_b * (0.5 * out["block_logvar"]).exp()
        grip_delta = grip_delta + eps_g * (0.5 * out["gripper_logvar"]).exp()
    block_delta = block_delta * block_scale
    grip_delta = grip_delta * grip_scale
    batch = state.shape[0]
    new = state.clone()
    new[:, POS] = state[:, POS] + block_delta[..., 0:3].reshape(batch, 9)
    new[:, ROT] = state[:, ROT] + block_delta[..., 3:9].reshape(batch, 18)
    new[:, VEL] = state[:, VEL] + block_delta[..., 9:15].reshape(batch, 18)
    new[:, 57:60] = state[:, 57:60] + grip_delta
    closed = state[:, 60].clone()
    closed = torch.where(action[:, 3] > 0.9, torch.ones_like(closed), closed)
    closed = torch.where(action[:, 3] < -0.9, torch.zeros_like(closed), closed)
    new[:, 60] = closed
    contacts = torch.zeros_like(state[:, CONTACT])
    per_block = torch.sigmoid(out["block_contact_logits"])
    for b in range(N_BLOCKS):
        contacts[:, SUPPORT[b][0]] = per_block[:, b, 0]
        contacts[:, SUPPORT[b][1]] = per_block[:, b, 1]
        contacts[:, ROBOT[b]] = per_block[:, b, 2]
    contacts[:, 0:3] = torch.sigmoid(out["pair_contact_logits"])
    new[:, CONTACT] = contacts
    return new


def rollout(model, state, actions, delta_scale, sample=False, noise=None, collect=None):
    """Integrate from `state` through every action. actions (B, T, 4) -> states (B, T, 61).

    With `collect` (a list), a hybrid model's gate probabilities are appended per step, so expert
    usage can be regularised and monitored during rollouts as well.
    """
    states = []
    current = state
    for t in range(actions.shape[1]):
        out = model(current, actions[:, t])
        if collect is not None and "gate_probabilities" in out:
            collect.append(out["gate_probabilities"])
        step_noise = None if noise is None else (noise[0][:, t], noise[1][:, t])
        current = apply_step(current, actions[:, t], out, delta_scale, sample, step_noise)
        states.append(current)
    return torch.stack(states, dim=1)


def change_magnitude(state, next_state, state_scale):
    """Size of a transition in the model's OWN normalised state (pose, rotation, velocity).

    Generic by construction: no task quantity, no notion of which object matters or what failure
    is. Used only to weight transitions.
    """
    return ((next_state[..., :45] - state[..., :45]) / state_scale[:45]).norm(dim=-1)


class TransitionWeights:
    """Square-root inverse-frequency weights over log change magnitude (fixed before training).

    87% of recorded steps barely move anything and ~0.5% are large, so an unweighted loss learns
    stillness. A 20-bin histogram of log10(magnitude) over TRAINING transitions gives each bin a
    frequency f; weight = 1/sqrt(f) for bins ABOVE the median change, and the median bin's weight
    for every bin at or below it -- so only rare LARGE changes are up-weighted, never rare tiny
    ones (near-zero jitter is also rare in log space, and boosting it would teach stillness).
    Normalised to mean 1 over the training transitions and capped at 20. The square root is the
    standard long-tail compromise: rare steps count more without a few extreme ones dominating.
    """

    BINS, CAP = 20, 20.0

    def __init__(self, magnitudes):
        logm = torch.log10(magnitudes.clamp_min(1e-12))
        self.low, self.high = float(logm.min()), float(logm.max())
        index = self._bin(logm)
        counts = torch.bincount(index, minlength=self.BINS).float()
        freq = counts / counts.sum()
        table = torch.where(counts > 0, 1.0 / freq.clamp_min(1e-12).sqrt(), torch.zeros_like(freq))
        median_bin = int(self._bin(logm.median().reshape(1))[0])
        table[:median_bin + 1] = table[median_bin]          # never up-weight small changes
        mean = float((table[index]).mean())
        self.table = (table / mean).clamp(max=self.CAP)

    def _bin(self, logm):
        span = max(self.high - self.low, 1e-9)
        return ((logm - self.low) / span * self.BINS).long().clamp(0, self.BINS - 1)

    def __call__(self, magnitudes):
        index = self._bin(torch.log10(magnitudes.clamp_min(1e-12)))
        return self.table.to(magnitudes.device)[index.to(self.table.device)].to(magnitudes.device)
