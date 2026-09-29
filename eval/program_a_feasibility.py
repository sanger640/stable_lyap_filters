"""Program A feasibility: causal re-anchoring, warning lead time and model-free pair selection.

Frozen protocol: results/jenga/program_a_feasibility_protocol.json.  Read-only: no simulation, no
training, and no task/failure label is read.  Every reference is the frozen physical monitor's own
decision on recorded physical trajectories.  DEV and TEST stay closed; from the benchmark file only
the shared execution-residual ``snippets`` array is read.

The 09-28 interface analysis re-anchored every probe to its OWN recorded state, which a robot can
never observe for counterfactual probes.  Here only the executed branch is observed:

* A1 (shared anchor): every probe restarts from the executed branch's recorded state.
* A2 (offset anchor): every probe restarts from that state plus its own predicted divergence.

T3 asks whether candidate pairs can be picked by a fixed action-space rule, T1 whether the
re-anchoring recovery survives A1/A2, and T2 (only if T1 passes) whether the alarm is early.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, nearest_cross_branch_pairs,  # noqa: E402
                                   smooth_vs_branch_alarm)
from jenga_d4_train import partition_agreement  # noqa: E402
from jenga_reobservation_interface import (binary_metrics, compare_nested,  # noqa: E402
                                           nested_stage_rows, range_summary,
                                           reanchored_rollout, sha256, validation_rows)
from jenga_w5_eval import load_model  # noqa: E402
from state_dynamics import orthonormalise_rotation, rollout  # noqa: E402


DATA = ROOT / "results/jenga/d4_neighborhood_data_large.npz"
NESTED = ROOT / "results/jenga/d3_data"
BENCH = ROOT / "results/jenga/bench/jenga_bench.npz"
TRACE_DATA = ROOT / "results/jenga/trace_data"
PROTOCOL = ROOT / "results/jenga/program_a_feasibility_protocol.json"
PRIOR = ROOT / "results/jenga/reobservation_interface_result.json"
OUTPUT = ROOT / "results/jenga/program_a_feasibility_result.json"

STEPS = 38
ACTION_STEPS = 8
INTERVALS = (1, 2, 4)
VARIANTS = ("A1", "A2")
CONFIG_SEEDS = (108, 109)
ROBUST_DRAWS = 8
RANDOM_DRAWS = 1000
BUDGETS = (3, 8)
NARROW = -1                                   # nested level index of the narrowest pair

ADDITIVE = np.r_[0:45, 57:60]                 # positions, 6D rotations, velocities, gripper xyz
ROTATION = slice(9, 27)
COPIED = np.r_[45:57, 60]                     # contact flags and gripper-closed flag


# --------------------------------------------------------------------------------------------
# Causal anchoring


def segment_boundaries(interval, stop=None, steps=STEPS):
    """Segment starts for re-anchoring every ``interval`` steps, optionally only up to ``stop``.

    With ``stop`` the last segment starts at ``stop`` and runs open-loop to the horizon, which is
    what a monitor that has observed ``stop`` executed steps can compute.
    """
    starts = list(range(0, steps, interval))
    if stop is None:
        return starts
    return sorted({b for b in starts if b < stop} | {stop})


def shared_anchor(truth, executed):
    """A1: every row restarts from its executed row's recorded state."""
    def anchor(boundary, _predicted):
        return truth[executed, boundary - 1]
    return anchor


def offset_anchor(truth, executed):
    """A2: executed recorded state plus the row's own predicted divergence from the executed row.

    Continuous dimensions are additive; each block's 6D rotation is re-orthonormalised; contact
    and gripper-closed flags are copied from the row's own prediction.
    """
    def anchor(boundary, predicted):
        own = predicted[:, boundary - 1]
        state = own.copy()
        state[:, ADDITIVE] = (truth[executed, boundary - 1][:, ADDITIVE] + own[:, ADDITIVE]
                              - predicted[executed, boundary - 1][:, ADDITIVE])
        state[:, ROTATION] = orthonormalise_rotation(
            torch.as_tensor(state[:, ROTATION])).numpy()
        state[:, COPIED] = own[:, COPIED]
        return state
    return anchor


def anchored_rollout(model, model_scale, starts, actions, anchor, boundaries, device,
                     batch_size=4096, rollout_function=rollout):
    """Roll out segment by segment; each later segment starts from ``anchor(boundary, predicted)``.

    Rows are coupled through the executed row, so a whole segment is predicted for every row
    before the next anchor is formed.
    """
    starts = np.asarray(starts, np.float32)
    actions = np.asarray(actions, np.float32)
    predicted = np.empty((*actions.shape[:2], starts.shape[1]), np.float32)
    boundaries = list(boundaries)
    if not boundaries or boundaries[0] != 0:
        raise ValueError("boundaries must start at 0")
    for number, first in enumerate(boundaries):
        stop = boundaries[number + 1] if number + 1 < len(boundaries) else actions.shape[1]
        segment_start = starts if first == 0 else np.asarray(anchor(first, predicted), np.float32)
        for begin in range(0, len(actions), batch_size):
            end = min(begin + batch_size, len(actions))
            state = torch.as_tensor(segment_start[begin:end], device=device)
            control = torch.as_tensor(actions[begin:end, first:stop], device=device)
            with torch.no_grad():
                values = rollout_function(model, state, control, model_scale)
            predicted[begin:end, first:stop] = values.cpu().numpy()
    return predicted


# --------------------------------------------------------------------------------------------
# T3: fixed action-space pairs


def fixed_pairs(snippets, budget):
    """(argmin, argmax) probe along each of the first ``budget`` principal snippet directions."""
    flat = np.asarray(snippets, float).reshape(len(snippets), -1)
    centered = flat - flat.mean(0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    pairs = []
    for direction in directions[:budget]:
        projection = centered @ direction
        pairs.append((int(np.argmin(projection)), int(np.argmax(projection))))
    return pairs


def covers(labels, pairs):
    labels = np.asarray(labels)
    return any(labels[left] != labels[right] for left, right in pairs)


def covering_distance(x, labels, pairs):
    """Nearest covering pair's action distance over the median nearest-neighbour probe spacing."""
    distance = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    spacing = float(np.median(distance.min(1)))
    crossing = [distance[a, b] for a, b in pairs if labels[a] != labels[b]]
    return float(min(crossing) / spacing) if crossing else None


def random_coverage(label_sets, budget, draws=RANDOM_DRAWS, seed=0, probes=64):
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(draws):
        hits = 0
        for labels in label_sets:
            pairs = []
            for _ in range(budget):
                left, right = rng.choice(probes, 2, replace=False)
                pairs.append((int(left), int(right)))
            hits += covers(labels, pairs)
        totals.append(hits / max(len(label_sets), 1))
    return float(np.mean(totals))


# --------------------------------------------------------------------------------------------
# Monitor evaluation (CPU pool)


def _init_worker():
    torch.set_num_threads(1)


def _candidate_state(job):
    """Frozen candidate decision and partition for one neighbourhood (None if it cannot run)."""
    control, trajectories = job
    try:
        evidence = smooth_vs_branch_alarm(control, trajectories[:, :18], trajectories)
        x, labels = action_branch_partition(control, trajectories[:, :18], trajectories)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    return bool(evidence.alarm), np.asarray(labels), np.asarray(x)


def evaluate_candidates(pool, actions, traces):
    jobs = [(control, trace[:, :, :45]) for control, trace in zip(actions, traces)]
    return list(pool.map(_candidate_state, jobs, chunksize=2))


def candidate_metrics(physical, predicted):
    """Identical to 09-28 candidate_rows: states where either side fails are skipped."""
    reference, decision, partitions = [], [], []
    for truth, estimate in zip(physical, predicted):
        if truth is None or estimate is None:
            continue
        reference.append(truth[0]); decision.append(estimate[0])
        partitions.append(partition_agreement(truth[1], estimate[1]))
    result = binary_metrics(reference, decision)
    result["valid_states"] = len(reference)
    result["mean_partition_agreement"] = float(np.mean(partitions)) if partitions else None
    return result


# --------------------------------------------------------------------------------------------
# T2: warning lead time


def split_index(physical_pair, state_scale):
    """First trace index where the narrow physical pair's separation reaches half its final value."""
    scale = np.maximum(np.asarray(state_scale, float)[:45], 1e-6)
    difference = (physical_pair[0, :, :45] - physical_pair[1, :, :45]) / scale
    separation = np.sqrt(np.mean(difference ** 2, axis=-1))
    reached = np.flatnonzero(separation >= .5 * separation[-1])
    return int(reached[0])


T2_KEYS = ("early_warning_recall", "fired_before_deadline_fraction", "negative_any_alarm_rate")


def endpoint_mean(by_endpoint, key):
    values = [e[key] for e in by_endpoint]
    return None if any(v is None for v in values) else float(np.mean(values))


def deadline(split, action_steps=ACTION_STEPS):
    """Executed-step count by which a warning must arrive: before the split or the action's end.

    A split at trace index i is observed after i + 1 executed steps, so an alarm at boundary t
    (t executed steps observed) is early iff t < min(i + 1, action_steps).
    """
    return min(split + 1, action_steps)


# --------------------------------------------------------------------------------------------
# Data


def source_identity(stem):
    episode, seed = stem.split("_")
    return episode[2:], int(seed[4:])


def load_nested(directory, keep):
    starts, actions, traces, sources = [], [], [], []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        episode, seed = source_identity(path.stem)
        if not keep(episode, seed):
            continue
        data = np.load(path, allow_pickle=False)
        if not len(data["start"]):
            continue
        starts.append(data["start"]); actions.append(data["actions"])
        traces.append(data["traces"]); sources.extend([path.stem] * len(data["start"]))
    return (np.concatenate(starts), np.concatenate(actions), np.concatenate(traces), sources)


def d2_fit_ids(directory, fraction=.12):
    """Episode:seed ids used to FIT D2 (jenga_w6_simple holds out the first 12% of sorted ids)."""
    ids = sorted({f"{e}:{s}" for e, s in (source_identity(p.stem)
                                          for p in Path(directory).glob("ep*_seed*.npz"))})
    held = set(ids[:max(1, int(len(ids) * fraction))])
    return set(ids) - held


def build_axis(name, source, keep_row, keep_nested):
    rows = np.flatnonzero([keep_row(e, s) for e, s in zip(source["episode_id"], source["seed"])])
    starts, actions, traces, sources = load_nested(NESTED, keep_nested)
    return {
        "name": name, "rows": rows,
        "episodes": source["episode_id"][rows], "seeds": source["seed"][rows],
        "start": source["start"][rows], "actions": source["actions"][rows],
        "traces": source["traces"][rows],
        "nested_start": starts, "nested_actions": actions, "nested_traces": traces,
        "nested_sources": sources,
    }


def flat_layout(axis):
    """The 09-28 layout: every neighbourhood probe, then every nested path."""
    actions, traces = axis["actions"], axis["traces"]
    n_actions, n_traces = axis["nested_actions"], axis["nested_traces"]
    start = np.concatenate([
        np.repeat(axis["start"][:, None], actions.shape[1], axis=1).reshape(-1, 61),
        np.repeat(axis["nested_start"][:, None], n_actions.shape[1] * 2, axis=1).reshape(-1, 61)])
    control = np.concatenate([actions.reshape(-1, STEPS, 4), n_actions.reshape(-1, STEPS, 4)])
    truth = np.concatenate([traces.reshape(-1, STEPS, 61), n_traces.reshape(-1, STEPS, 61)])
    return start, control, truth, len(axis["start"]) * actions.shape[1]


def neighbourhood_executed(states, probes, probe):
    return (np.arange(states)[:, None] * probes + probe).repeat(probes, axis=1).reshape(-1)


def nested_executed(groups, levels, endpoint):
    paths = levels * 2
    executed = np.arange(groups) * paths + (levels - 1) * 2 + endpoint
    return np.repeat(executed, paths)


# --------------------------------------------------------------------------------------------
# Aggregation and decisions


def median_of(values):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return float(np.median(finite)) if len(finite) else None


def recovery(condition, open_loop, oracle):
    denominator = oracle - open_loop
    return float((condition - open_loop) / denominator) if denominator > 0 else float("nan")


def t1_axis_summary(per_seed):
    """Median over seeds of every gate quantity for one axis."""
    open_candidate = median_of([s["open_loop"]["candidate"]["recall"] for s in per_seed])
    open_added = median_of([s["open_loop"]["nested_final_added"] for s in per_seed])
    summary = {"open_loop": {"candidate_recall": open_candidate,
                             "final_recall": median_of([s["open_loop"]["nested_final_recall"]
                                                        for s in per_seed]),
                             "final_added_positive": open_added},
               "conditions": {}}
    for interval in INTERVALS:
        oracle = f"oracle_{interval}"
        summary["conditions"][oracle] = {
            "candidate_recall": median_of([s[oracle]["candidate"]["recall"] for s in per_seed]),
            "final_recall": median_of([s[oracle]["nested_final_recall"] for s in per_seed]),
            "final_added_positive": median_of([s[oracle]["nested_final_added"]
                                               for s in per_seed])}
        for variant in VARIANTS:
            key = f"{variant}_{interval}"
            row = {
                "candidate_recall": median_of([s[key]["candidate"]["recall"] for s in per_seed]),
                "candidate_recall_robust": median_of([s[key]["candidate_robust_recall"]
                                                      for s in per_seed]),
                "final_recall": median_of([s[key]["nested_final_recall"] for s in per_seed]),
                "final_added_positive": median_of([s[key]["nested_final_added"]
                                                   for s in per_seed]),
                "recovery": median_of([recovery(s[key]["nested_final_recall"],
                                                s["open_loop"]["nested_final_recall"],
                                                s[oracle]["nested_final_recall"])
                                       for s in per_seed]),
                "recovery_undefined_seeds": int(sum(
                    not np.isfinite(recovery(s[key]["nested_final_recall"],
                                             s["open_loop"]["nested_final_recall"],
                                             s[oracle]["nested_final_recall"]))
                    for s in per_seed)),
            }
            row["passes"] = bool(
                row["recovery"] is not None and row["recovery"] >= .5
                and row["final_added_positive"] <= open_added + .05
                and row["candidate_recall"] >= open_candidate - .05)
            summary["conditions"][key] = row
    return summary


def t1_decision(axis_summaries):
    passing = [(interval, variant) for interval in INTERVALS for variant in VARIANTS
               if all(axis_summaries[a]["conditions"][f"{variant}_{interval}"]["passes"]
                      for a in ("episode", "configuration"))]
    if not passing:
        return {"passes": False, "selected": None, "passing": []}
    interval = max(i for i, _ in passing)
    variant = "A1" if (interval, "A1") in passing else "A2"
    return {"passes": True, "selected": {"variant": variant, "interval": interval},
            "passing": [f"{v}_{i}" for i, v in passing]}


def t3_decision(axis_results):
    checks = {}
    for axis in ("episode", "configuration"):
        row = axis_results[axis]["budgets"]["8"]
        checks[axis] = bool(row["fixed"] >= .75 and row["fixed"] >= row["random"] + .15
                            and row["fixed"] >= axis_results[axis]["d2_open_loop"])
    return {"per_axis": checks, "passes": all(checks.values())}


def overall_decision(t1, t2, t3):
    if not t1["passes"]:
        return "T1_fail"
    if not t2["passes"]:
        return "T1_pass_T2_fail"
    return "T1_pass_T2_pass_T3_pass" if t3["passes"] else "T1_pass_T2_pass_T3_fail"


# --------------------------------------------------------------------------------------------


def nested_final(physical_rows, predicted_rows):
    stages = compare_nested(physical_rows, predicted_rows)
    return stages["alarm"]["recall"], stages["alarm"]["added_positive_rate"], stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--models", nargs="+", default=[
        str(ROOT / f"results/jenga/w6_cw_d2_s{seed}.pt") for seed in range(1, 11)])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; use --force deliberately")
    protocol = json.loads(PROTOCOL.read_text())
    if not protocol.get("status", "").startswith("frozen before execution"):
        raise SystemExit("protocol is not frozen")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    prior = json.loads(PRIOR.read_text())

    source = np.load(DATA, allow_pickle=False)
    source = {key: source[key] for key in source.files}
    _, held_episodes = validation_rows(source["episode_id"])
    if list(held_episodes) != prior["heldout_episodes"]:
        raise SystemExit("episode axis differs from the 09-28 analysis")
    axes = {
        "episode": build_axis("episode", source, lambda e, s: e in held_episodes,
                              lambda e, s: e in held_episodes),
        "configuration": build_axis("configuration", source, lambda e, s: s in CONFIG_SEEDS,
                                    lambda e, s: s in CONFIG_SEEDS),
    }
    with np.load(BENCH, allow_pickle=False) as bench:
        snippets = bench["snippets"]                 # the ONLY key read from the benchmark file
    for axis in axes.values():
        residual = axis["actions"][:, :1, :ACTION_STEPS, :3] - axis["actions"][:, :, :ACTION_STEPS, :3]
        error = float(np.abs(residual - (snippets[None] - snippets[:1][None])).max())
        if error > 1e-5:
            raise SystemExit(f"probe/snippet mapping broken on {axis['name']}: {error}")
        axis["snippet_mapping_error"] = error
    primary = int(np.argmin(np.linalg.norm(snippets.reshape(len(snippets), -1), axis=1)))
    robust = [int(p) for p in np.random.default_rng(0).choice(len(snippets), ROBUST_DRAWS,
                                                               replace=False)]
    fit_ids = d2_fit_ids(TRACE_DATA)

    def in_fit(episodes, seeds):
        return float(np.mean([f"{e}:{s}" in fit_ids for e, s in zip(episodes, seeds)]))

    overlap = {name: {
        "neighbourhood_states": int(len(a["rows"])),
        "nested_groups": int(len(a["nested_start"])),
        "neighbourhood_fraction_in_d2_fit": in_fit(a["episodes"], a["seeds"]),
        "nested_fraction_in_d2_fit": in_fit(*zip(*(source_identity(s)
                                                   for s in a["nested_sources"]))),
    } for name, a in axes.items()}
    joint_rows = sorted(set(axes["episode"]["rows"]) & set(axes["configuration"]["rows"]))
    joint_nested = [s for s in axes["episode"]["nested_sources"]
                    if source_identity(s)[1] in CONFIG_SEEDS]

    pool = ProcessPoolExecutor(args.workers, initializer=_init_worker)
    physical = {}
    for name, axis in axes.items():
        physical[name] = evaluate_candidates(pool, axis["actions"], axis["traces"])
        print(f"physical candidates {name}: {sum(p is not None and p[0] for p in physical[name])}"
              f"/{len(physical[name])}", flush=True)

    # ---------------- Phase 1: reproduce the 09-28 open-loop and per-branch oracle -------------
    models = []
    reproduction = {"checked": [], "matches": True}
    rollouts = {}
    for number, model_path in enumerate(args.models):
        model, model_scale = load_model(model_path, device)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_scale = checkpoint["state_scale"].cpu().numpy()
        models.append({"path": model_path, "sha256": sha256(model_path),
                       "seed": checkpoint.get("seed"), "state_scale": state_scale})
        for name, axis in axes.items():
            start, control, truth, count = flat_layout(axis)
            for interval in (STEPS, *INTERVALS):
                predicted = reanchored_rollout(model, model_scale, start, control, truth,
                                               interval, device, args.batch_size)
                rollouts[(number, name, interval)] = (predicted[:count].reshape(axis["traces"].shape),
                                                      predicted[count:].reshape(
                                                          axis["nested_traces"].shape))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"phase 1 rollouts model {number + 1}/{len(args.models)}", flush=True)

    candidates = {}
    nested_rows = {}
    for number, info in enumerate(models):
        for name, axis in axes.items():
            nested_rows[(number, name, "physical")] = nested_stage_rows(
                axis["nested_actions"], axis["nested_traces"], info["state_scale"])
            for interval in (STEPS, *INTERVALS):
                neighbourhood, nested = rollouts[(number, name, interval)]
                candidates[(number, name, interval)] = evaluate_candidates(
                    pool, axis["actions"], neighbourhood)
                nested_rows[(number, name, interval)] = nested_stage_rows(
                    axis["nested_actions"], nested, info["state_scale"])
        for interval in (STEPS, *INTERVALS):
            prior_row = prior["per_seed"][number]["intervals"][str(interval)]
            mine = {"candidate": candidate_metrics(physical["episode"],
                                                   candidates[(number, "episode", interval)]),
                    "nested": compare_nested(nested_rows[(number, "episode", "physical")],
                                             nested_rows[(number, "episode", interval)])}
            match = (prior["per_seed"][number]["model_sha256"] == info["sha256"]
                     and mine["candidate"] == prior_row["candidate"]
                     and mine["nested"] == prior_row["nested"])
            reproduction["checked"].append({"seed": info["seed"], "interval": interval,
                                            "match": bool(match)})
            reproduction["matches"] &= bool(match)
        print(f"phase 1 monitor model {number + 1}/{len(models)}", flush=True)
    if not reproduction["matches"]:
        bad = [row for row in reproduction["checked"] if not row["match"]]
        output.with_suffix(".reproduction_failure.json").write_text(
            json.dumps(reproduction, indent=2) + "\n")
        raise SystemExit(f"09-28 reproduction failed; no new condition read: {bad[:5]}")
    print("phase 1: 09-28 open-loop and per-branch oracle reproduced exactly", flush=True)

    # ---------------- T3: fixed action-space pairs --------------------------------------------
    t3 = {"primary_probe": primary, "pairs": {str(b): fixed_pairs(snippets, b) for b in BUDGETS},
          "axes": {}}
    subsets = {"episode": None, "configuration": None, "joint": joint_rows}
    for name in ("episode", "configuration", "joint"):
        base = "episode" if name == "joint" else name
        keep = (range(len(axes[base]["rows"])) if subsets[name] is None else
                [i for i, r in enumerate(axes[base]["rows"]) if r in set(joint_rows)])
        states = [i for i in keep if physical[base][i] is not None and physical[base][i][0]]
        labels = [physical[base][i][1] for i in states]
        result = {"physical_candidate_states": len(states), "budgets": {}}
        for budget in BUDGETS:
            pairs = t3["pairs"][str(budget)]
            fixed = float(np.mean([covers(l, pairs) for l in labels])) if labels else None
            distances = [covering_distance(physical[base][i][2], physical[base][i][1], pairs)
                         for i in states]
            result["budgets"][str(budget)] = {
                "fixed": fixed,
                "random": random_coverage(labels, budget) if labels else None,
                "covering_distance_over_spacing_median": median_of(
                    [d for d in distances if d is not None]),
            }
        d2 = []
        for number in range(len(models)):
            hits = []
            for i in states:
                estimate = candidates[(number, base, STEPS)][i]
                try:
                    pairs = nearest_cross_branch_pairs(estimate[2], estimate[1], 3)
                    hits.append(covers(physical[base][i][1], pairs))
                except (ValueError, TypeError):
                    hits.append(False)
            d2.append(float(np.mean(hits)) if hits else None)
        result["d2_open_loop_per_seed"] = d2
        result["d2_open_loop"] = median_of(d2)
        t3["axes"][name] = result
    t3["decision"] = t3_decision(t3["axes"])
    print(f"T3: {json.dumps(t3['decision'])}", flush=True)

    # ---------------- T1: causal re-anchoring ------------------------------------------------
    t1_per_seed = {name: [] for name in axes}
    for number, info in enumerate(models):
        model, model_scale = load_model(info["path"], device)
        for name, axis in axes.items():
            states, probes = axis["actions"].shape[:2]
            groups, levels = axis["nested_actions"].shape[:2]
            n_start = np.repeat(axis["start"][:, None], probes, 1).reshape(-1, 61)
            n_actions = axis["actions"].reshape(-1, STEPS, 4)
            n_truth = axis["traces"].reshape(-1, STEPS, 61)
            g_start = np.repeat(axis["nested_start"][:, None], levels * 2, 1).reshape(-1, 61)
            g_actions = axis["nested_actions"].reshape(-1, STEPS, 4)
            g_truth = axis["nested_traces"].reshape(-1, STEPS, 61)
            physical_nested = nested_rows[(number, name, "physical")]
            row = {"open_loop": None}
            for interval in (STEPS, *INTERVALS):
                key = "open_loop" if interval == STEPS else f"oracle_{interval}"
                recall, added, stages = nested_final(physical_nested,
                                                     nested_rows[(number, name, interval)])
                row[key] = {"candidate": candidate_metrics(physical[name],
                                                           candidates[(number, name, interval)]),
                            "nested": stages, "nested_final_recall": recall,
                            "nested_final_added": added}
            for interval in INTERVALS:
                boundaries = segment_boundaries(interval)
                for variant in VARIANTS:
                    make = shared_anchor if variant == "A1" else offset_anchor
                    robust_recalls, primary_candidate = [], None
                    for probe in [primary, *robust]:
                        executed = neighbourhood_executed(states, probes, probe)
                        predicted = anchored_rollout(model, model_scale, n_start, n_actions,
                                                     make(n_truth, executed), boundaries,
                                                     device, args.batch_size)
                        metrics = candidate_metrics(physical[name], evaluate_candidates(
                            pool, axis["actions"], predicted.reshape(axis["traces"].shape)))
                        if probe == primary and primary_candidate is None:
                            primary_candidate = metrics
                        else:
                            robust_recalls.append(metrics["recall"])
                    endpoint_rows = []
                    for endpoint in (0, 1):
                        executed = nested_executed(groups, levels, endpoint)
                        predicted = anchored_rollout(model, model_scale, g_start, g_actions,
                                                     make(g_truth, executed), boundaries,
                                                     device, args.batch_size)
                        endpoint_rows.append(nested_final(physical_nested, nested_stage_rows(
                            axis["nested_actions"], predicted.reshape(axis["nested_traces"].shape),
                            info["state_scale"])))
                    row[f"{variant}_{interval}"] = {
                        "candidate": primary_candidate,
                        "candidate_robust_recall": median_of(robust_recalls),
                        "candidate_robust_recalls": robust_recalls,
                        "nested_by_executed_endpoint": [r[2] for r in endpoint_rows],
                        "nested_final_recall": float(np.mean([r[0] for r in endpoint_rows])),
                        "nested_final_added": float(np.mean([r[1] for r in endpoint_rows])),
                    }
            row["seed"] = info["seed"]
            t1_per_seed[name].append(row)
            print(f"T1 model {number + 1}/{len(models)} axis {name}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t1 = {"per_seed": t1_per_seed,
          "summary": {name: t1_axis_summary(rows) for name, rows in t1_per_seed.items()}}
    t1["decision"] = t1_decision(t1["summary"])
    print(f"T1: {json.dumps(t1['decision'])}", flush=True)

    # ---------------- T2: warning lead time (only if T1 passes) ------------------------------
    t2 = {"ran": False, "passes": False}
    if t1["decision"]["passes"]:
        variant = t1["decision"]["selected"]["variant"]
        interval = t1["decision"]["selected"]["interval"]
        make = shared_anchor if variant == "A1" else offset_anchor
        t2 = {"ran": True, "variant": variant, "interval": interval, "axes": {}}
        for name, axis in axes.items():
            groups, levels = axis["nested_actions"].shape[:2]
            g_start = np.repeat(axis["nested_start"][:, None], levels * 2, 1).reshape(-1, 61)
            g_actions = axis["nested_actions"].reshape(-1, STEPS, 4)
            g_truth = axis["nested_traces"].reshape(-1, STEPS, 61)
            per_seed = []
            for number, info in enumerate(models):
                model, model_scale = load_model(info["path"], device)
                physical_nested = nested_rows[(number, name, "physical")]
                positive = [r["alarm"] for r in physical_nested]
                splits = [split_index(group[NARROW], info["state_scale"]) if p else None
                          for group, p in zip(axis["nested_traces"], positive)]
                by_endpoint = []
                for endpoint in (0, 1):
                    executed = nested_executed(groups, levels, endpoint)
                    first_alarm = [None] * groups
                    for stop in range(0, STEPS, interval):
                        predicted = anchored_rollout(
                            model, model_scale, g_start, g_actions, make(g_truth, executed),
                            segment_boundaries(interval, stop), device, args.batch_size)
                        decisions = nested_stage_rows(
                            axis["nested_actions"], predicted.reshape(axis["nested_traces"].shape),
                            info["state_scale"])
                        for g, decision in enumerate(decisions):
                            if decision["alarm"] and first_alarm[g] is None:
                                first_alarm[g] = stop
                    early = [first_alarm[g] is not None and first_alarm[g] < deadline(splits[g])
                             for g in range(groups) if positive[g]]
                    fired = [g for g in range(groups) if positive[g] and first_alarm[g] is not None]
                    negatives = [g for g in range(groups) if not positive[g]]
                    by_endpoint.append({
                        "early_warning_recall": float(np.mean(early)) if early else None,
                        "fired_before_deadline_fraction": float(np.mean(
                            [first_alarm[g] < deadline(splits[g]) for g in fired])) if fired else None,
                        "negative_any_alarm_rate": float(np.mean(
                            [first_alarm[g] is not None for g in negatives])) if negatives else None,
                        "lead_steps": [deadline(splits[g]) - first_alarm[g] for g in fired],
                        "first_alarm": first_alarm, "split_index": splits,
                    })
                # Protocol: report both executed endpoints and gate on their mean.
                per_seed.append({"seed": info["seed"], "by_executed_endpoint": by_endpoint,
                                 **{key: endpoint_mean(by_endpoint, key) for key in T2_KEYS}})
                del model
                print(f"T2 model {number + 1}/{len(models)} axis {name}", flush=True)
            summary = {key: median_of([s[key] for s in per_seed]) for key in T2_KEYS}
            summary["passes"] = bool(
                summary["early_warning_recall"] is not None
                and summary["early_warning_recall"] >= .40
                and summary["fired_before_deadline_fraction"] is not None
                and summary["fired_before_deadline_fraction"] >= .50)
            t2["axes"][name] = {"per_seed": per_seed, "summary": summary}
        t2["passes"] = all(t2["axes"][a]["summary"]["passes"] for a in ("episode", "configuration"))
        print(f"T2: passes={t2['passes']}", flush=True)
    pool.shutdown()

    decision = overall_decision(t1["decision"], t2, t3["decision"])
    result = {
        "protocol": protocol,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(PROTOCOL),
        "data_sha256": sha256(DATA),
        "bench_sha256_snippets_only_read": sha256(BENCH),
        "prior_result_sha256": sha256(PRIOR),
        "models": [{k: v for k, v in m.items() if k != "state_scale"} for m in models],
        "axes": overlap,
        "joint": {"neighbourhood_states": len(joint_rows), "nested_groups": len(joint_nested),
                  "note": "joint rows are included in both axes; T3 also reports them alone"},
        "snippet_mapping_error": {n: a["snippet_mapping_error"] for n, a in axes.items()},
        "executed_probe": {"primary": primary, "robust": robust},
        "reproduction": reproduction,
        "T3": t3, "T1": t1, "T2": t2,
        "decision": {"outcome": decision, "action": protocol["decision"][decision]},
    }
    output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({"T3": t3["decision"], "T1": t1["decision"],
                      "T1_summary": t1["summary"],
                      "T2": {a: t2["axes"][a]["summary"] for a in t2.get("axes", {})},
                      "decision": result["decision"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
