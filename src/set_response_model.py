"""Direct task-free response models for counterfactual action sets.

Both arms predict a complete normalized physical response without autoregressive state feedback.
The independent arm processes probe tokens separately.  The set-conditioned arm inserts
permutation-equivariant self-attention before the shared decoder, allowing the response to be
interpreted relative to the complete local action neighborhood.
"""
import torch
import torch.nn as nn


STATE_DIM = 61
ACTION_DIM = 4
RESPONSE_DIM = 45


def _mlp(input_dim, output_dim, hidden):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, output_dim),
    )


class DirectSetResponseModel(nn.Module):
    """Map ``(state, unordered action set)`` directly to normalized response trajectories."""

    def __init__(self, steps=38, hidden=128, heads=4, layers=2, set_conditioned=True):
        super().__init__()
        self.steps = int(steps)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.layers = int(layers)
        self.set_conditioned = bool(set_conditioned)
        self.state_encoder = _mlp(STATE_DIM, hidden, hidden)
        self.action_encoder = _mlp(self.steps * ACTION_DIM, hidden, hidden)
        self.token_encoder = _mlp(2 * hidden, hidden, hidden)
        if self.set_conditioned:
            layer = nn.TransformerEncoderLayer(
                hidden, heads, dim_feedforward=4 * hidden, dropout=0., activation="gelu",
                batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.context = nn.Sequential(*[
                nn.Sequential(nn.LayerNorm(hidden), _mlp(hidden, hidden, hidden))
                for _ in range(layers)
            ])
        self.output_norm = nn.LayerNorm(hidden)
        self.decoder = _mlp(hidden, self.steps * RESPONSE_DIM, hidden)

    def _independent_context(self, tokens):
        values = tokens
        for block in self.context:
            values = values + block(values)
        return values

    def encode_tokens(self, state, actions, state_mean, state_scale, action_mean, action_scale):
        """Encode an unordered action set into permutation-equivariant response tokens."""
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError("state must have shape (batch, 61)")
        if actions.ndim != 4 or actions.shape[0] != len(state) or actions.shape[2:] != (
                self.steps, ACTION_DIM):
            raise ValueError("actions must have shape (batch, probes, steps, 4)")
        state_z = (state - state_mean) / state_scale
        action_z = (actions - action_mean) / action_scale
        scene = self.state_encoder(state_z)[:, None].expand(-1, actions.shape[1], -1)
        control = self.action_encoder(action_z.flatten(2))
        tokens = self.token_encoder(torch.cat([scene, control], dim=-1))
        return self.context(tokens) if self.set_conditioned else self._independent_context(tokens)

    def forward(self, state, actions, state_mean, state_scale, action_mean, action_scale):
        """Return normalized response deltas with shape ``(B, probes, steps, 45)``."""
        values = self.encode_tokens(
            state, actions, state_mean, state_scale, action_mean, action_scale)
        return self.decoder(self.output_norm(values)).reshape(
            len(state), actions.shape[1], self.steps, RESPONSE_DIM)


class NestedPairCurveHead(nn.Module):
    """Predict task-free boundary scalars and full separation curves for nested action pairs.

    The output is in ``log1p`` space. Pair features are symmetric in their two endpoints, while
    every endpoint token remains conditioned on the complete unordered set of nested probes.
    """

    def __init__(self, backbone, curve_steps=39, hidden=128):
        super().__init__()
        self.backbone = backbone
        self.curve_steps = int(curve_steps)
        token_dim = backbone.hidden
        self.head = _mlp(3 * token_dim, 2 + self.curve_steps, hidden)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        if pair_actions.ndim != 5 or pair_actions.shape[2] != 2:
            raise ValueError("pair_actions must have shape (batch, levels, 2, steps, 4)")
        batch, levels = pair_actions.shape[:2]
        flat = pair_actions.reshape(batch, 2 * levels, pair_actions.shape[-2], 4)
        tokens = self.backbone.encode_tokens(
            state, flat, state_mean, state_scale, action_mean, action_scale)
        pairs = tokens.reshape(batch, levels, 2, -1)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        symmetric = torch.cat([(left + right) * .5, torch.abs(left - right), left * right], dim=-1)
        return self.head(symmetric)


class DirectTemporalPairCurveModel(nn.Module):
    """Predict nested separation curves directly from scene state and raw paired controls."""

    def __init__(self, steps=38, action_horizon=8, curve_steps=39, hidden=128, heads=4, layers=2):
        super().__init__()
        self.steps = int(steps)
        self.action_horizon = int(action_horizon)
        self.curve_steps = int(curve_steps)
        self.hidden = int(hidden)
        self.state_encoder = _mlp(STATE_DIM, hidden, hidden)
        self.action_encoder = nn.GRU(ACTION_DIM, hidden, batch_first=True)
        self.pair_encoder = _mlp(4 * hidden, hidden, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=4 * hidden, dropout=0., activation="gelu",
            batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = _mlp(hidden, 2 + self.curve_steps, hidden)

    def encode_pairs(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        """Return endpoint-symmetric, level-equivariant pair features."""
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError("state must have shape (batch, 61)")
        if (pair_actions.ndim != 5 or pair_actions.shape[0] != len(state)
                or pair_actions.shape[2:] != (2, self.steps, ACTION_DIM)):
            raise ValueError("pair_actions must have shape (batch, levels, 2, steps, 4)")
        batch, levels = pair_actions.shape[:2]
        scene = self.state_encoder((state - state_mean) / state_scale)
        control = ((pair_actions[..., :self.action_horizon, :] -
                    action_mean[:self.action_horizon]) / action_scale[:self.action_horizon])
        flat = control.reshape(batch * levels * 2, self.action_horizon, ACTION_DIM)
        _, encoded = self.action_encoder(flat)
        endpoints = encoded[-1].reshape(batch, levels, 2, self.hidden)
        left, right = endpoints[:, :, 0], endpoints[:, :, 1]
        pair = torch.cat([scene[:, None].expand(-1, levels, -1),
                          (left + right) * .5, torch.abs(left - right), left * right], dim=-1)
        return self.context(self.pair_encoder(pair))

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        return self.head(values)


class AmplitudeShapeTemporalCurveModel(DirectTemporalPairCurveModel):
    """Factor nonnegative log-separation into exact-zero amplitude and normalized shape."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.head
        self.amplitude_head = _mlp(self.hidden, 1, self.hidden)
        self.shape_head = _mlp(self.hidden, 2 + self.curve_steps, self.hidden)
        # Start every sample in the live ReLU region; continuous zero-amplitude supervision may
        # then drive only truly zero-response examples across the exact structural boundary.
        with torch.no_grad():
            self.amplitude_head[-1].bias.fill_(0.5)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        amplitude = torch.relu(self.amplitude_head(values))
        shape = torch.nn.functional.softplus(self.shape_head(values))
        return amplitude * shape, amplitude, shape


class GroupAmplitudeShapeTemporalCurveModel(DirectTemporalPairCurveModel):
    """Use one exact-zero response amplitude for a complete nested refinement group.

    Level features are mean pooled before amplitude prediction, making the response/no-response
    decision invariant to level ordering.  Per-level features still predict the normalized curve
    shape, so refinement-specific geometry is retained whenever the shared amplitude is positive.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.head
        self.amplitude_head = _mlp(self.hidden, 1, self.hidden)
        self.shape_head = _mlp(self.hidden, 2 + self.curve_steps, self.hidden)
        with torch.no_grad():
            self.amplitude_head[-1].bias.fill_(0.5)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        amplitude = torch.relu(self.amplitude_head(values.mean(dim=1)))[:, None, :]
        shape = torch.nn.functional.softplus(self.shape_head(values))
        return amplitude * shape, amplitude, shape


class DirectMonitorEvidenceModel(DirectTemporalPairCurveModel):
    """Predict normalized continuous monitor statistics for a refinement group."""

    def __init__(self, evidence_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.evidence_dim = int(evidence_dim)
        del self.head
        self.evidence_head = _mlp(2 * self.hidden, self.evidence_dim, self.hidden)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        pooled = torch.cat([values.mean(dim=1), values.max(dim=1).values], dim=-1)
        return self.evidence_head(pooled)


class JointCurveMonitorEvidenceModel(DirectTemporalPairCurveModel):
    """Jointly predict D9b curves and continuous monitor evidence from one encoder."""

    def __init__(self, evidence_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.evidence_dim = int(evidence_dim)
        del self.head
        self.amplitude_head = _mlp(self.hidden, 1, self.hidden)
        self.shape_head = _mlp(self.hidden, 2 + self.curve_steps, self.hidden)
        self.evidence_head = _mlp(2 * self.hidden, self.evidence_dim, self.hidden)
        with torch.no_grad():
            self.amplitude_head[-1].bias.fill_(0.5)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        amplitude = torch.relu(self.amplitude_head(values.mean(dim=1)))[:, None, :]
        shape = torch.nn.functional.softplus(self.shape_head(values))
        pooled = torch.cat([values.mean(dim=1), values.max(dim=1).values], dim=-1)
        evidence = self.evidence_head(pooled)
        return amplitude * shape, amplitude, shape, evidence


class ObjectGraphJointMonitorEvidenceModel(JointCurveMonitorEvidenceModel):
    """D13 joint head with an anonymous, permutation-invariant object/contact graph.

    The three Jenga blocks share every weight.  Nodes contain only generic rigid-body
    quantities and per-body contact flags; directed edges contain relative geometry and the
    corresponding pair-contact flag.  Mean/max pooling removes block ordering before the same
    action-pair, curve, and evidence heads used by :class:`JointCurveMonitorEvidenceModel`.

    Absolute vertical height is retained because gravity/support breaks vertical translation
    symmetry.  Horizontal positions appear only relative to the object centroid and gripper, so
    the representation is invariant to a joint horizontal translation of scene and controls.
    """

    N_OBJECTS = 3
    POSITION_SCALE = .05
    VELOCITY_SCALE = .10
    PAIRS = {(0, 1): 0, (0, 2): 1, (1, 2): 2}

    def __init__(self, graph_rounds=3, **kwargs):
        super().__init__(**kwargs)
        del self.state_encoder
        hidden = self.hidden
        # centred xyz, gripper-relative xyz, height, rotation6, velocity6,
        # table/floor/robot contact = 22 generic per-object channels.
        self.object_encoder = _mlp(22, hidden, hidden)
        # relative xyz, distance, pair contact.
        self.edge_encoder = _mlp(5, hidden, hidden)
        self.edge_updates = nn.ModuleList(
            [_mlp(3 * hidden, hidden, hidden) for _ in range(graph_rounds)])
        self.node_updates = nn.ModuleList(
            [_mlp(2 * hidden, hidden, hidden) for _ in range(graph_rounds)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(graph_rounds)])
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(graph_rounds)])
        # gripper relative xyz, height, closed flag.
        self.gripper_encoder = _mlp(5, hidden, hidden)
        self.scene_fusion = _mlp(3 * hidden, hidden, hidden)

    @classmethod
    def graph_inputs(cls, state):
        """Parse the 61-D state into generic object, edge, and gripper features."""
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError("state must have shape (batch, 61)")
        batch = len(state)
        position = state[:, :9].reshape(batch, cls.N_OBJECTS, 3)
        rotation = state[:, 9:27].reshape(batch, cls.N_OBJECTS, 6)
        velocity = state[:, 27:45].reshape(batch, cls.N_OBJECTS, 6)
        contacts = state[:, 45:57]
        gripper = state[:, 57:61]
        centroid = position.mean(dim=1, keepdim=True)
        own_contacts = torch.stack([
            torch.stack([contacts[:, 3 + block], contacts[:, 6 + block],
                         contacts[:, 9 + block]], dim=-1)
            for block in range(cls.N_OBJECTS)], dim=1)
        objects = torch.cat([
            (position - centroid) / cls.POSITION_SCALE,
            (position - gripper[:, None, :3]) / cls.POSITION_SCALE,
            position[..., 2:3] / cls.POSITION_SCALE,
            rotation,
            velocity / cls.VELOCITY_SCALE,
            own_contacts,
        ], dim=-1)
        senders, receivers, edges = [], [], []
        for sender in range(cls.N_OBJECTS):
            for receiver in range(cls.N_OBJECTS):
                if sender == receiver:
                    continue
                relative = ((position[:, receiver] - position[:, sender]) /
                            cls.POSITION_SCALE)
                contact = contacts[:, cls.PAIRS[tuple(sorted((sender, receiver)))]]
                edges.append(torch.cat(
                    [relative, relative.norm(dim=-1, keepdim=True), contact[:, None]], dim=-1))
                senders.append(sender); receivers.append(receiver)
        grip = torch.cat([
            (gripper[:, :3] - centroid[:, 0]) / cls.POSITION_SCALE,
            gripper[:, 2:3] / cls.POSITION_SCALE,
            gripper[:, 3:4],
        ], dim=-1)
        return (objects, torch.stack(edges, dim=1), grip,
                torch.tensor(senders, device=state.device),
                torch.tensor(receivers, device=state.device))

    def encode_scene(self, state):
        objects, edges, gripper, senders, receivers = self.graph_inputs(state)
        nodes = self.object_encoder(objects)
        edge_values = self.edge_encoder(edges)
        for edge_update, node_update, edge_norm, node_norm in zip(
                self.edge_updates, self.node_updates, self.edge_norms, self.node_norms):
            edge_values = edge_norm(edge_values + edge_update(torch.cat([
                edge_values, nodes[:, senders], nodes[:, receivers]], dim=-1)))
            incoming = torch.zeros_like(nodes).index_add_(1, receivers, edge_values)
            nodes = node_norm(nodes + node_update(torch.cat([nodes, incoming], dim=-1)))
        pooled = torch.cat([
            nodes.mean(dim=1), nodes.max(dim=1).values, self.gripper_encoder(gripper)], dim=-1)
        return self.scene_fusion(pooled)

    def encode_pairs(self, state, pair_actions, state_mean, state_scale,
                     action_mean, action_scale):
        # state_mean/state_scale remain in the interface to make this a drop-in D13 control;
        # the graph uses shared physical units so object-specific normalizers cannot leak labels.
        del state_mean, state_scale
        if (pair_actions.ndim != 5 or pair_actions.shape[0] != len(state)
                or pair_actions.shape[2:] != (2, self.steps, ACTION_DIM)):
            raise ValueError("pair_actions must have shape (batch, levels, 2, steps, 4)")
        batch, levels = pair_actions.shape[:2]
        scene = self.encode_scene(state)
        # Cartesian targets are represented relative to the current gripper.  This is the
        # control analogue of relative graph geometry and prevents world-frame translation from
        # becoming a spurious cue.  The gripper command remains normalized in its native units.
        control = pair_actions[..., :self.action_horizon, :].clone()
        control[..., :3] = control[..., :3] - state[:, None, None, None, 57:60]
        control[..., :3] = control[..., :3] / action_scale[:self.action_horizon, :3]
        control[..., 3:] = ((control[..., 3:] - action_mean[:self.action_horizon, 3:]) /
                            action_scale[:self.action_horizon, 3:])
        flat = control.reshape(batch * levels * 2, self.action_horizon, ACTION_DIM)
        _, encoded = self.action_encoder(flat)
        endpoints = encoded[-1].reshape(batch, levels, 2, self.hidden)
        left, right = endpoints[:, :, 0], endpoints[:, :, 1]
        pair = torch.cat([scene[:, None].expand(-1, levels, -1),
                          (left + right) * .5, torch.abs(left - right), left * right], dim=-1)
        return self.context(self.pair_encoder(pair))


class HistoryGroupAmplitudeShapeTemporalCurveModel(GroupAmplitudeShapeTemporalCurveModel):
    """D9b curve model conditioned on a short, generic pre-probe state/action history.

    The history encoder receives only the same task-free physical state and executed-control
    channels available to the snapshot model.  Controls are aligned with their destination state:
    the first history state receives a zero control token and each later state receives the control
    that produced it.  Probe endpoint and level symmetries are unchanged from D9b.
    """

    def __init__(self, history_states=4, **kwargs):
        super().__init__(**kwargs)
        self.history_states = int(history_states)
        del self.state_encoder
        self.history_encoder = nn.GRU(STATE_DIM + ACTION_DIM, self.hidden, batch_first=True)

    def encode_pairs(self, state_history, history_actions, pair_actions,
                     history_state_mean, history_state_scale,
                     history_action_mean, history_action_scale,
                     action_mean, action_scale):
        if (state_history.ndim != 3 or state_history.shape[1:] !=
                (self.history_states, STATE_DIM)):
            raise ValueError("state_history must have shape (batch, history_states, 61)")
        if (history_actions.ndim != 3 or history_actions.shape !=
                (len(state_history), self.history_states - 1, ACTION_DIM)):
            raise ValueError("history_actions must align consecutive history states")
        if (pair_actions.ndim != 5 or pair_actions.shape[0] != len(state_history)
                or pair_actions.shape[2:] != (2, self.steps, ACTION_DIM)):
            raise ValueError("pair_actions must have shape (batch, levels, 2, steps, 4)")

        state_z = (state_history - history_state_mean) / history_state_scale
        action_z = (history_actions - history_action_mean) / history_action_scale
        leading_zero = torch.zeros_like(action_z[:, :1])
        aligned_actions = torch.cat([leading_zero, action_z], dim=1)
        _, scene_encoded = self.history_encoder(torch.cat([state_z, aligned_actions], dim=-1))
        scene = scene_encoded[-1]

        batch, levels = pair_actions.shape[:2]
        control = ((pair_actions[..., :self.action_horizon, :] -
                    action_mean[:self.action_horizon]) / action_scale[:self.action_horizon])
        flat = control.reshape(batch * levels * 2, self.action_horizon, ACTION_DIM)
        _, encoded = self.action_encoder(flat)
        endpoints = encoded[-1].reshape(batch, levels, 2, self.hidden)
        left, right = endpoints[:, :, 0], endpoints[:, :, 1]
        pair = torch.cat([scene[:, None].expand(-1, levels, -1),
                          (left + right) * .5, torch.abs(left - right), left * right], dim=-1)
        return self.context(self.pair_encoder(pair))

    def forward(self, state_history, history_actions, pair_actions,
                history_state_mean, history_state_scale,
                history_action_mean, history_action_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state_history, history_actions, pair_actions,
            history_state_mean, history_state_scale,
            history_action_mean, history_action_scale, action_mean, action_scale)
        amplitude = torch.relu(self.amplitude_head(values.mean(dim=1)))[:, None, :]
        shape = torch.nn.functional.softplus(self.shape_head(values))
        return amplitude * shape, amplitude, shape


class InteractionEventGroupAmplitudeShapeCurveModel(GroupAmplitudeShapeTemporalCurveModel):
    """D9b conditioned on explicit, task-independent recent interaction events.

    Each transition token contains signed and absolute continuous-state change, contact presence,
    contact creation/loss, and the executed control.  Shared token encoding followed by mean, max,
    and latest-event pooling does not assign semantic mode names or task-specific object roles.  The
    current-state and future-action paths otherwise match D9b.
    """

    EVENT_DIM = 2 * RESPONSE_DIM + 4 * 12 + ACTION_DIM

    def __init__(self, history_states=4, **kwargs):
        super().__init__(**kwargs)
        self.history_states = int(history_states)
        self.event_encoder = _mlp(self.EVENT_DIM, self.hidden, self.hidden)
        self.scene_fusion = _mlp(4 * self.hidden, self.hidden, self.hidden)

    def event_features(self, state_history, history_actions,
                       delta_mean, delta_scale, history_action_mean, history_action_scale):
        if (state_history.ndim != 3 or state_history.shape[1:] !=
                (self.history_states, STATE_DIM)):
            raise ValueError("state_history must have shape (batch, history_states, 61)")
        if (history_actions.ndim != 3 or history_actions.shape !=
                (len(state_history), self.history_states - 1, ACTION_DIM)):
            raise ValueError("history_actions must align consecutive history states")
        continuous_delta = state_history[:, 1:, :RESPONSE_DIM] - state_history[:, :-1, :RESPONSE_DIM]
        signed = (continuous_delta - delta_mean) / delta_scale
        magnitude = continuous_delta.abs() / delta_scale
        before = state_history[:, :-1, 45:57]
        after = state_history[:, 1:, 45:57]
        created = torch.relu(after - before)
        lost = torch.relu(before - after)
        control = (history_actions - history_action_mean) / history_action_scale
        return torch.cat([signed, magnitude, before, after, created, lost, control], dim=-1)

    def encode_pairs(self, state_history, history_actions, pair_actions,
                     state_mean, state_scale, delta_mean, delta_scale,
                     history_action_mean, history_action_scale, action_mean, action_scale):
        if (pair_actions.ndim != 5 or pair_actions.shape[0] != len(state_history)
                or pair_actions.shape[2:] != (2, self.steps, ACTION_DIM)):
            raise ValueError("pair_actions must have shape (batch, levels, 2, steps, 4)")
        current = self.state_encoder((state_history[:, -1] - state_mean) / state_scale)
        events = self.event_encoder(self.event_features(
            state_history, history_actions, delta_mean, delta_scale,
            history_action_mean, history_action_scale))
        scene = self.scene_fusion(torch.cat(
            [current, events.mean(dim=1), events.max(dim=1).values, events[:, -1]], dim=-1))

        batch, levels = pair_actions.shape[:2]
        control = ((pair_actions[..., :self.action_horizon, :] -
                    action_mean[:self.action_horizon]) / action_scale[:self.action_horizon])
        flat = control.reshape(batch * levels * 2, self.action_horizon, ACTION_DIM)
        _, encoded = self.action_encoder(flat)
        endpoints = encoded[-1].reshape(batch, levels, 2, self.hidden)
        left, right = endpoints[:, :, 0], endpoints[:, :, 1]
        pair = torch.cat([scene[:, None].expand(-1, levels, -1),
                          (left + right) * .5, torch.abs(left - right), left * right], dim=-1)
        return self.context(self.pair_encoder(pair))

    def forward(self, state_history, history_actions, pair_actions,
                state_mean, state_scale, delta_mean, delta_scale,
                history_action_mean, history_action_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state_history, history_actions, pair_actions, state_mean, state_scale,
            delta_mean, delta_scale, history_action_mean, history_action_scale,
            action_mean, action_scale)
        amplitude = torch.relu(self.amplitude_head(values.mean(dim=1)))[:, None, :]
        shape = torch.nn.functional.softplus(self.shape_head(values))
        return amplitude * shape, amplitude, shape


class MixtureGroupAmplitudeShapeCurveModel(DirectTemporalPairCurveModel):
    """A shared categorical distribution over complete nonnegative nested response curves.

    Every latent component predicts one group amplitude and one shape for every refinement level.
    The component logits are pooled over levels, so one component is selected for the complete
    group rather than independently per output. Endpoint-swap and level-permutation symmetries are
    inherited from the D9b pair encoder.
    """

    def __init__(self, modes=3, **kwargs):
        super().__init__(**kwargs)
        if modes < 2:
            raise ValueError("curve mixture needs at least two modes")
        self.modes = int(modes)
        del self.head
        self.mode_head = _mlp(self.hidden, self.modes, self.hidden)
        self.amplitude_head = _mlp(self.hidden, self.modes, self.hidden)
        self.shape_head = _mlp(
            self.hidden, self.modes * (2 + self.curve_steps), self.hidden)
        with torch.no_grad():
            self.amplitude_head[-1].bias.fill_(0.5)

    def forward(self, state, pair_actions, state_mean, state_scale, action_mean, action_scale):
        values = self.encode_pairs(
            state, pair_actions, state_mean, state_scale, action_mean, action_scale)
        pooled = values.mean(dim=1)
        logits = self.mode_head(pooled)
        amplitude = torch.relu(self.amplitude_head(pooled))
        shape = torch.nn.functional.softplus(self.shape_head(values)).reshape(
            len(state), pair_actions.shape[1], self.modes, 2 + self.curve_steps)
        curves = amplitude[:, None, :, None] * shape
        return curves.permute(0, 2, 1, 3), logits, amplitude


class SnapshotCurveModeRouter(nn.Module):
    """Route among frozen curve modes from permutation-invariant snapshot pair features."""

    def __init__(self, hidden=128, modes=3):
        super().__init__()
        self.classifier = _mlp(2 * hidden, modes, hidden)

    def forward(self, pair_features):
        if pair_features.ndim != 3:
            raise ValueError("pair_features must have shape (batch, levels, hidden)")
        return self.classifier(torch.cat(
            [pair_features.mean(dim=1), pair_features.max(dim=1).values], dim=-1))


class EventCurveModeRouter(nn.Module):
    """Route frozen curve modes using snapshot features plus generic transition events."""

    def __init__(self, event_dim=142, hidden=128, modes=3):
        super().__init__()
        self.event_encoder = _mlp(event_dim, hidden, hidden)
        self.classifier = _mlp(5 * hidden, modes, hidden)

    def forward(self, pair_features, event_features):
        if pair_features.ndim != 3 or event_features.ndim != 3:
            raise ValueError("pair/event features must have shapes (B,L,H)/(B,T,E)")
        events = self.event_encoder(event_features)
        values = torch.cat([pair_features.mean(dim=1), pair_features.max(dim=1).values,
                            events.mean(dim=1), events.max(dim=1).values, events[:, -1]], dim=-1)
        return self.classifier(values)


def reconstruct_states(normalized_delta, start, truth_template, response_scale):
    """Restore first-45 state fields; retain known robot/contact channels from the template."""
    predicted = truth_template.clone()
    predicted[..., :RESPONSE_DIM] = (
        start[:, None, None, :RESPONSE_DIM]
        + normalized_delta * response_scale[:RESPONSE_DIM])
    return predicted
