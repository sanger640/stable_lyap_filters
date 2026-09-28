"""Calibration-free action-conditioned smooth-versus-branch monitor.

Compare one smooth response surface ``action error -> future trajectory`` with two independent
smooth surfaces on regions of the same action neighbourhood.  Model evidence and description-
length charges for searching over splits and fragmenting the local action graph decide the alarm;
no reference/calibration states or task labels enter.
"""
from dataclasses import asdict, dataclass

import numpy as np

MIN_REGION = 2
ACTION_DIMS = 5
TRAJECTORY_DIMS = 6
LENGTH_MULTIPLIERS = (0.5, 1.0, 2.0)
NOISE_FRACTIONS = (1e-3, 1e-2, 1e-1)


@dataclass(frozen=True)
class BranchEvidence:
    alarm: bool
    early_delta_description_length: float
    full_delta_description_length: float
    region_sizes: tuple
    action_graph_k: int
    action_connected: bool
    action_cut_edges: int
    partition_description_length: float


@dataclass(frozen=True)
class PredictiveBranchEvidence:
    """Out-of-sample comparison; lower prediction error is better."""
    alarm: bool
    early_smooth_mse: float
    early_branch_mse: float
    full_smooth_mse: float
    full_branch_mse: float
    early_relative_improvement: float
    full_relative_improvement: float
    early_fold_wins: int
    full_fold_wins: int
    folds: int


@dataclass(frozen=True)
class BoundaryRefinementEvidence:
    """Scaling-law evidence from repeatedly bisecting candidate action boundaries."""
    alarm: bool
    early_branch_pairs: int
    full_branch_pairs: int
    pairs: int
    refinements: int
    early_delta_bic: tuple
    full_delta_bic: tuple


@dataclass(frozen=True)
class DynamicsBranchEvidence:
    """Evidence that nearby actions obey different post-action evolution laws."""
    alarm: bool
    persistent_branch_pairs: int
    pairs: int
    early_branch_pairs: int
    late_branch_pairs: int
    early_delta_evidence: tuple
    late_delta_evidence: tuple


def _standardise_actions(action_errors):
    x = np.asarray(action_errors, float)
    if x.ndim < 2:
        raise ValueError("action_errors must have shape (probes, ...)")
    x = x.reshape(len(x), -1)
    x = x - x.mean(0)
    scale = x.std(0)
    keep = scale > 1e-10
    if not np.any(keep):
        raise ValueError("action probes do not vary")
    x = x[:, keep] / scale[keep]
    # Execution snippets are correlated over time. Work in their leading intrinsic directions so
    # a 64-point sample is not treated as a well-covered 24-D cube.
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    z = (u * s)[:, :min(ACTION_DIMS, len(s))]
    return z / np.maximum(z.std(0), 1e-10)


def _trajectory_coordinates(trajectories, dims=TRAJECTORY_DIMS):
    y = np.asarray(trajectories, float)
    if y.ndim < 3:
        raise ValueError("trajectories must have shape (probes, time, features...)")
    y = y.reshape(len(y), -1)
    y = y - y.mean(0)
    u, s, _ = np.linalg.svd(y, full_matrices=False)
    z = (u * s)[:, :min(dims, len(s))]
    keep = z.std(0) > 1e-10
    if not np.any(keep):
        return np.zeros((len(y), 1))
    z = z[:, keep]
    # One common scale preserves how much variance each trajectory PC explains. Per-PC whitening
    # would make the leading direction arbitrary and erase precisely the dominant branch signal.
    return z / max(float(np.sqrt(np.mean(z ** 2))), 1e-10)


def _rbf(x, length):
    distance2 = np.sum((x[:, None] - x[None]) ** 2, axis=2)
    return np.exp(-0.5 * distance2 / max(length * length, 1e-12))


def _median_distance(x):
    distance = np.sqrt(np.sum((x[:, None] - x[None]) ** 2, axis=2))
    upper = distance[np.triu_indices(len(x), 1)]
    positive = upper[upper > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def _log_evidence(y, signal, noise):
    """Independent zero-mean GP evidence for each trajectory coordinate."""
    n, outputs = y.shape
    covariance = signal + (noise * noise + 1e-8) * np.eye(n)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros_like(y)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    evidence = (-0.5 * np.sum(y * alpha) - 0.5 * outputs * logdet
                - 0.5 * n * outputs * np.log(2 * np.pi))
    posterior = signal @ alpha
    return float(evidence), posterior


def _best_smooth(x, y, labels=None):
    median = _median_distance(x)
    best = (-np.inf, None)
    for multiplier in LENGTH_MULTIPLIERS:
        signal = _rbf(x, median * multiplier)
        if labels is not None:
            signal = signal * (labels[:, None] == labels[None, :])
        for noise in NOISE_FRACTIONS:
            evidence, posterior = _log_evidence(y, signal, noise)
            if evidence > best[0]:
                best = (evidence, posterior)
    return best


def _best_predict(x_train, y_train, x_test, labels=None, test_labels=None):
    """Select GP hyperparameters on training evidence, then predict unseen action probes."""
    median = _median_distance(x_train)
    best = (-np.inf, None)
    for multiplier in LENGTH_MULTIPLIERS:
        length = median * multiplier
        signal = _rbf(x_train, length)
        if labels is not None:
            signal = signal * (labels[:, None] == labels[None, :])
        cross = np.exp(-0.5 * np.sum(
            (x_test[:, None] - x_train[None, :]) ** 2, axis=2) / max(length * length, 1e-12))
        if labels is not None:
            cross = cross * (test_labels[:, None] == labels[None, :])
        for noise in NOISE_FRACTIONS:
            evidence, _ = _log_evidence(y_train, signal, noise)
            if evidence <= best[0]:
                continue
            covariance = signal + (noise * noise + 1e-8) * np.eye(len(x_train))
            try:
                alpha = np.linalg.solve(covariance, y_train)
            except np.linalg.LinAlgError:
                continue
            best = (evidence, cross @ alpha)
    if best[1] is None:
        raise RuntimeError("no finite GP fit")
    return best


def _train_test_trajectory_coordinates(train, test, dims=TRAJECTORY_DIMS):
    """Fit trajectory PCA only on visible probes, then project the hidden probes."""
    train = np.asarray(train, float).reshape(len(train), -1)
    test = np.asarray(test, float).reshape(len(test), -1)
    mean = train.mean(0)
    centred = train - mean
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:min(dims, len(axes))]
    train_z = centred @ axes.T
    test_z = (test - mean) @ axes.T
    keep = train_z.std(0) > 1e-10
    if not np.any(keep):
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))
    train_z, test_z = train_z[:, keep], test_z[:, keep]
    scale = max(float(np.sqrt(np.mean(train_z ** 2))), 1e-10)
    return train_z / scale, test_z / scale


def _unique_candidates(candidates):
    return [labels for number, labels in enumerate(candidates)
            if not any(np.array_equal(labels, earlier) or np.array_equal(labels, 1 - earlier)
                       for earlier in candidates[:number])]


def _select_training_partition(x, early, full):
    """Choose a partition using visible trajectories only."""
    _, prediction = _best_smooth(x, full)
    candidates = _unique_candidates([
        _candidate_partition(full), _candidate_partition(full - prediction)])
    graph, _ = _minimal_connected_graph(x)
    scored = []
    for labels in candidates:
        partition_cost, _, _ = _partition_description_length(graph, labels)
        early_gain = _description_gain(x, early, labels) - partition_cost
        full_gain = _description_gain(x, full, labels) - partition_cost
        scored.append((min(early_gain, full_gain), labels))
    return max(scored, key=lambda item: item[0])[1]


def _action_knn_labels(x_train, labels, x_test, neighbours=5):
    """Assign hidden probes from action location alone; trajectories are unavailable here."""
    distance2 = np.sum((x_test[:, None] - x_train[None, :]) ** 2, axis=2)
    k = min(neighbours, len(x_train))
    if k % 2 == 0:
        k -= 1
    near = np.argsort(distance2, axis=1)[:, :max(k, 1)]
    return (np.mean(labels[near], axis=1) > 0.5).astype(int)


def _fold_ids(x, folds):
    """Deterministic action-balanced folds; every fold spans the leading action direction."""
    if folds < 2 or folds > len(x) // (2 * MIN_REGION):
        raise ValueError("invalid fold count")
    order = np.argsort(x[:, 0], kind="stable")
    ids = np.empty(len(x), int)
    ids[order] = np.arange(len(x)) % folds
    return ids


def action_branch_partition(action_errors, early_trajectories, full_trajectories):
    """Return the action coordinates and selected partition used by the evidence monitor."""
    x = _standardise_actions(action_errors)
    early = _trajectory_coordinates(early_trajectories)
    full = _trajectory_coordinates(full_trajectories)
    if len(x) != len(early) or len(x) != len(full):
        raise ValueError("actions and trajectories must have the same probe count")
    return x, _select_training_partition(x, early, full)


def nearest_cross_branch_pairs(action_coordinates, labels, count=3):
    """Greedily choose nearby opposite-branch pairs without reusing endpoints."""
    x = np.asarray(action_coordinates, float)
    labels = np.asarray(labels, int)
    candidates = []
    for left in np.flatnonzero(labels == 0):
        for right in np.flatnonzero(labels == 1):
            candidates.append((float(np.linalg.norm(x[left] - x[right])), int(left), int(right)))
    candidates.sort()
    selected, used = [], set()
    for _, left, right in candidates:
        if left in used or right in used:
            continue
        selected.append((left, right)); used.update((left, right))
        if len(selected) == count:
            break
    if len(selected) < count:
        for _, left, right in candidates:
            if (left, right) not in selected:
                selected.append((left, right))
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError("not enough cross-branch action pairs")
    return selected


def _scaling_bic(gaps, plateau):
    """Fit local gap as linear/quadratic in bracket width; add an intercept for a branch."""
    from scipy.optimize import least_squares

    gaps = np.asarray(gaps, float)
    width = .5 ** np.arange(len(gaps), dtype=float)
    if plateau:
        def residual(params):
            offset, linear, quadratic = params
            return gaps - (offset + linear * width + quadratic * width ** 2)
        fit = least_squares(residual, x0=[max(gaps[-1], 0), gaps[0] - gaps[-1], 0],
                            bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
        parameters = 3
    else:
        def residual(params):
            linear, quadratic = params
            return gaps - (linear * width + quadratic * width ** 2)
        fit = least_squares(residual, x0=[gaps[0], 0])
        parameters = 2
    rss = float(np.sum(fit.fun ** 2))
    n = len(gaps)
    floor = 1e-12 * max(float(np.mean(gaps ** 2)), 1e-12)
    return float(n * np.log(max(rss / n, floor)) + parameters * np.log(n))


def boundary_refinement_alarm(early_gaps, full_gaps):
    """Alarm when most refined brackets support a persistent gap at both horizons.

    Arrays have shape (candidate pairs, initial bracket + refinement levels). The decision is the
    sign of BIC evidence and a fixed majority over the independently refined boundary brackets.
    """
    early = np.asarray(early_gaps, float)
    full = np.asarray(full_gaps, float)
    if early.shape != full.shape or early.ndim != 2 or early.shape[1] < 4:
        raise ValueError("gap arrays must have matching shape (pairs, at least four levels)")
    early_delta = tuple(float(_scaling_bic(row, False) - _scaling_bic(row, True))
                        for row in early)
    full_delta = tuple(float(_scaling_bic(row, False) - _scaling_bic(row, True))
                       for row in full)
    needed = len(early) // 2 + 1
    early_pairs = sum(value > 0 for value in early_delta)
    full_pairs = sum(value > 0 for value in full_delta)
    alarm = early_pairs >= needed and full_pairs >= needed
    return BoundaryRefinementEvidence(bool(alarm), early_pairs, full_pairs, len(early),
                                      early.shape[1] - 1, early_delta, full_delta)


def _dynamics_transitions(pair, start, stop, dimensions=5):
    """Build pooled transition data after removing each trajectory's static pose offset.

    ``pair`` contains two dense pose+velocity traces. Positions use the block length as a unit;
    rotations are already dimensionless.  Each branch's pose at the action endpoint is removed,
    while velocity is retained because it is part of the instantaneous dynamical state.
    A pooled PCA supplies compact, anonymous coordinates without task labels or reference states.
    """
    traces = np.asarray(pair, float)
    if traces.ndim != 3 or traces.shape[0] != 2 or traces.shape[2] < 45:
        raise ValueError("each pair must have shape (2, time, at least 45 pose/velocity fields)")
    if not 0 <= start < stop < traces.shape[1]:
        raise ValueError("invalid dynamics interval")
    pose = traces[:, :, :27].copy()
    pose[:, :, :9] /= .05
    pose -= pose[:, start:start + 1]
    state = np.concatenate([pose, traces[:, :, 27:45]], axis=2)
    segment = state[:, start:stop + 1]
    current = segment[:, :-1].reshape(-1, state.shape[2])
    change = np.diff(segment, axis=1).reshape(-1, state.shape[2])

    # Fit both coordinate systems on the pooled branches. One common scale per system preserves
    # the dominant physical variation instead of whitening simulator-level numerical jitter.
    def coordinates(values, count):
        centred = values - values.mean(0)
        _, singular, axes = np.linalg.svd(centred, full_matrices=False)
        keep = singular > max(singular[0] if len(singular) else 0, 1.) * 1e-10
        axes = axes[:min(count, int(np.sum(keep)))]
        if not len(axes):
            return np.zeros((len(values), 1))
        result = centred @ axes.T
        return result / max(float(np.sqrt(np.mean(result ** 2))), 1e-10)

    x = coordinates(current, dimensions)
    y = coordinates(change, dimensions)
    # The held command is autonomous but finite-horizon transients can still depend smoothly on
    # phase. Supplying phase prevents that ordinary shared non-stationarity becoming a branch.
    phase = np.tile(np.linspace(0., 1., stop - start, endpoint=False), 2)[:, None]
    x = np.concatenate([x, phase], axis=1)
    labels = np.repeat(np.arange(2), stop - start)
    return x, y, labels


def _dynamics_evidence(pair, start, stop):
    """GP marginal-evidence gain for two laws over one shared nonlinear transition law."""
    x, y, labels = _dynamics_transitions(pair, start, stop)
    shared, _ = _best_smooth(x, y)
    split, _ = _best_smooth(x, y, labels)
    # Splitting introduces one binary regime variable. This BIC-style description charge prevents
    # accepting a second law for a negligible likelihood gain; zero remains the decision boundary.
    return float(split - shared - .5 * np.log(max(len(x), 2)))


def shared_vs_branch_dynamics_alarm(pair_traces, action_end=7, early_hold=10,
                                    full_hold=30):
    """Alarm only when most refined action brackets need two persistent dynamics laws.

    The early comparison uses hold steps 1..``early_hold`` and the late comparison uses the
    disjoint hold steps ``early_hold+1..full_hold``. Static branch-specific pose offsets are removed
    before fitting. Contacts and task outcomes are deliberately absent from the decision.
    """
    pairs = np.asarray(pair_traces, float)
    if pairs.ndim != 4 or pairs.shape[1] != 2:
        raise ValueError("pair_traces must have shape (pairs, 2, time, features)")
    if action_end + full_hold >= pairs.shape[2]:
        raise ValueError("traces do not cover the requested hold")
    early = tuple(_dynamics_evidence(pair, action_end, action_end + early_hold)
                  for pair in pairs)
    late = tuple(_dynamics_evidence(pair, action_end + early_hold,
                                    action_end + full_hold) for pair in pairs)
    early_pairs = sum(value > 0 for value in early)
    late_pairs = sum(value > 0 for value in late)
    persistent = sum(a > 0 and b > 0 for a, b in zip(early, late))
    needed = len(pairs) // 2 + 1
    return DynamicsBranchEvidence(bool(persistent >= needed), persistent, len(pairs),
                                  early_pairs, late_pairs, early, late)


def _candidate_partition(residual):
    centred = residual - residual.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    values = centred @ axis
    order = np.argsort(values)
    best, split = -np.inf, MIN_REGION
    n = len(values)
    for index in range(MIN_REGION, n - MIN_REGION + 1):
        left, right = values[order[:index]], values[order[index:]]
        between = index * (n - index) / n * (left.mean() - right.mean()) ** 2
        if between > best:
            best, split = between, index
    labels = np.zeros(n, int)
    labels[order[split:]] = 1
    return labels


def _minimal_connected_graph(x):
    distance = np.sqrt(np.sum((x[:, None] - x[None]) ** 2, axis=2))
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1)
    n = len(x)
    for k in range(1, n):
        graph = np.zeros((n, n), bool)
        rows = np.repeat(np.arange(n), k)
        graph[rows, order[:, :k].reshape(-1)] = True
        graph |= graph.T
        seen, stack = {0}, [0]
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(graph[node]):
                if int(nxt) not in seen:
                    seen.add(int(nxt)); stack.append(int(nxt))
        if len(seen) == n:
            return graph, k
    raise RuntimeError("could not connect action graph")


def _component_count(graph, labels, group):
    remaining = set(np.flatnonzero(labels == group).tolist())
    count = 0
    while remaining:
        count += 1
        start = next(iter(remaining)); remaining.remove(start)
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(graph[node]):
                nxt = int(nxt)
                if nxt in remaining and labels[nxt] == group:
                    remaining.remove(nxt); stack.append(nxt)
    return count


def _partition_description_length(graph, labels):
    """Graph-Potts code: simple boundaries cost less than interleaved/action-disconnected ones."""
    edges = np.transpose(np.triu(graph, 1).nonzero())
    cut = sum(labels[i] != labels[j] for i, j in edges)
    components = _component_count(graph, labels, 0) + _component_count(graph, labels, 1)
    # Encoding each boundary edge or extra component among |E| possibilities costs log|E| nats.
    cost = (cut + max(components - 2, 0)) * np.log(max(len(edges), 2))
    return float(cost), int(cut), int(components)


def _description_gain(x, y, labels, candidate_families=2):
    smooth, _ = _best_smooth(x, y)
    branch, _ = _best_smooth(x, y, labels)
    # The residual-PC1 candidate searches n-2*MIN_REGION+1 ordered split positions.
    search_penalty = np.log(max(candidate_families * (len(x) - 2 * MIN_REGION + 1), 1))
    return float(branch - smooth - search_penalty)


def smooth_vs_branch_alarm(action_errors, early_trajectories, full_trajectories):
    """Compare one smooth action-response surface against two persistent branch surfaces.

    The candidate partition is selected once from the full trajectory's smooth-model residual and
    then held fixed. Fragmented regions pay a description-length cost on the automatically built
    action-neighbour graph. Evidence must remain positive at both the early and full horizons.
    """
    x = _standardise_actions(action_errors)
    early = _trajectory_coordinates(early_trajectories)
    full = _trajectory_coordinates(full_trajectories)
    if len(x) != len(early) or len(x) != len(full):
        raise ValueError("actions and trajectories must have the same probe count")
    _, prediction = _best_smooth(x, full)
    candidates = [_candidate_partition(full), _candidate_partition(full - prediction)]
    graph, k = _minimal_connected_graph(x)
    candidates = _unique_candidates(candidates)
    scored = []
    for labels in candidates:
        partition_cost, cut, components = _partition_description_length(graph, labels)
        early_gain = _description_gain(x, early, labels) - partition_cost
        full_gain = _description_gain(x, full, labels) - partition_cost
        scored.append((min(early_gain, full_gain), labels, early_gain, full_gain,
                       partition_cost, cut, components))
    (_, labels, early_gain, full_gain, partition_cost, cut,
     components) = max(scored, key=lambda item: item[0])
    sizes = tuple(int(np.sum(labels == group)) for group in (0, 1))
    alarm = early_gain > 0 and full_gain > 0
    return BranchEvidence(bool(alarm), early_gain, full_gain, sizes, int(k),
                          bool(components == 2), cut, partition_cost)


def cross_validated_smooth_vs_branch_alarm(action_errors, early_trajectories,
                                            full_trajectories, folds=8):
    """Require an action-derived branch to predict trajectories that were hidden during fitting.

    In each fold, candidate discovery, trajectory PCA, GP fitting and hyperparameter selection use
    visible probes only. Hidden probes are assigned to a branch by action-space nearest neighbours;
    their trajectories enter only when scoring the already-made predictions. The branch alarm has
    no fitted threshold: its total squared prediction error must be lower at both horizons.
    """
    x = _standardise_actions(action_errors)
    early_raw = np.asarray(early_trajectories, float)
    full_raw = np.asarray(full_trajectories, float)
    if len(x) != len(early_raw) or len(x) != len(full_raw):
        raise ValueError("actions and trajectories must have the same probe count")
    fold_ids = _fold_ids(x, folds)
    errors = {name: [] for name in ("early_smooth", "early_branch",
                                     "full_smooth", "full_branch")}
    fold_wins = {"early": 0, "full": 0}
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        early_train, early_test = _train_test_trajectory_coordinates(
            early_raw[train], early_raw[test])
        full_train, full_test = _train_test_trajectory_coordinates(
            full_raw[train], full_raw[test])
        labels = _select_training_partition(x[train], early_train, full_train)
        test_labels = _action_knn_labels(x[train], labels, x[test])
        for horizon, y_train, y_test in (
                ("early", early_train, early_test), ("full", full_train, full_test)):
            _, smooth = _best_predict(x[train], y_train, x[test])
            _, branch = _best_predict(x[train], y_train, x[test], labels, test_labels)
            smooth_error = float(np.sum((y_test - smooth) ** 2))
            branch_error = float(np.sum((y_test - branch) ** 2))
            errors[f"{horizon}_smooth"].append(smooth_error)
            errors[f"{horizon}_branch"].append(branch_error)
            fold_wins[horizon] += int(branch_error < smooth_error)
    totals = {name: float(np.sum(value)) for name, value in errors.items()}
    early_improvement = ((totals["early_smooth"] - totals["early_branch"])
                         / max(totals["early_smooth"], 1e-12))
    full_improvement = ((totals["full_smooth"] - totals["full_branch"])
                        / max(totals["full_smooth"], 1e-12))
    alarm = early_improvement > 0 and full_improvement > 0
    return PredictiveBranchEvidence(
        bool(alarm), totals["early_smooth"], totals["early_branch"],
        totals["full_smooth"], totals["full_branch"], float(early_improvement),
        float(full_improvement), fold_wins["early"], fold_wins["full"], int(folds))


def result_dict(result):
    return asdict(result)
