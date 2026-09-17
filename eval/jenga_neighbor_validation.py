"""Frozen neighbor-topple validation, supported-boundary benchmark, and WM audit.

The quiet set estimates false positives.  The separately constructed boundary set
is screened using MuJoCo outcomes and estimates sensitivity only; it is not a
prevalence sample.  Detector settings and latent scaling are frozen from the
original local-probe experiment.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay, load_world_model  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import (coherent_probe_chunks, model_probe_deltas,
                                      probe_scalars, read_probe_cache,
                                      true_probe_deltas)  # noqa: E402
from jenga_ordered_change_points import (bootstrap_confidence, confusion,
                                          score_responses, summarise)  # noqa: E402
from jenga_ordered_holdout import binomial_interval  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, TOPPLE_DEG,
                                    extract_sim, projection_basis)  # noqa: E402

TAIL = 5
DETECTOR = {"dimension": 4, "degree": 3, "min_side": 8,
            "min_jump_ratio": 4.0}


def prior_keys(*cache_paths):
    keys = set()
    for path in cache_paths:
        if not path or not Path(path).exists():
            continue
        data = dict(np.load(path, allow_pickle=False))
        metadata = json.loads(str(data["metadata_json"]))
        keys.update((str(row["episode_id"]), int(row["chunk_start"])) for row in metadata)
    return keys


def spread_by_episode(indices, count, episode_ids, chunk_starts):
    """Round-robin episodes, taking temporal quantiles within each episode."""
    groups = {}
    for i in indices:
        groups.setdefault(str(episode_ids[i]), []).append(int(i))
    for episode in groups:
        groups[episode].sort(key=lambda i: int(chunk_starts[i]))
    ordered_episodes = sorted(groups, key=int)
    selected = []
    rank = 0
    while len(selected) < int(count):
        added = False
        for episode in ordered_episodes:
            values = groups[episode]
            if rank < len(values):
                # Alternate from early/late toward the middle to cover time.
                position = rank // 2 if rank % 2 == 0 else len(values) - 1 - rank // 2
                if 0 <= position < len(values):
                    selected.append(values[position]); added = True
                    if len(selected) == int(count):
                        break
        if not added:
            break
        rank += 1
    if len(selected) < int(count):
        raise ValueError(f"requested {count} rows from only {len(selected)} candidates")
    return selected


def choose_pools(scan_cache, excluded, quiet_count, candidate_count):
    data = dict(np.load(scan_cache, allow_pickle=False))
    ids = data["episode_ids"].astype(str); starts = data["chunk_starts"].astype(int)
    start_neighbor = np.max(data["start_tilt"][:, 1:], axis=1)
    peak_neighbor = np.max(data["tail5_peak_tilt"][:, 1:], axis=1)
    available = np.asarray([(e, int(s)) not in excluded for e, s in zip(ids, starts)])
    quiet_mask = available & (start_neighbor < TOPPLE_DEG) & (peak_neighbor < 5.0)
    quiet_indices = spread_by_episode(np.flatnonzero(quiet_mask), quiet_count, ids, starts)
    quiet_keys = {(ids[i], int(starts[i])) for i in quiet_indices}

    eligible = np.flatnonzero(available & (start_neighbor < TOPPLE_DEG)
                              & np.asarray([(e, int(s)) not in quiet_keys
                                            for e, s in zip(ids, starts)]))
    # Nominal outcomes close to 45 degrees are the highest-yield place to find a
    # probe-induced switch.  Episode/start tie-breaks keep this deterministic.
    ranked = sorted(eligible, key=lambda i: (abs(float(peak_neighbor[i]) - TOPPLE_DEG),
                                             int(ids[i]), int(starts[i])))
    candidate_indices = ranked[:min(int(candidate_count), len(ranked))]
    quiet = [{"episode_id": ids[i], "chunk_start": int(starts[i]),
              "stratum": "quiet_control", "screen_peak_neighbor_tail5": float(peak_neighbor[i])}
             for i in quiet_indices]
    candidates = [{"episode_id": ids[i], "chunk_start": int(starts[i]),
                   "stratum": "boundary_candidate",
                   "screen_peak_neighbor_tail5": float(peak_neighbor[i]),
                   "candidate_rank": rank}
                  for rank, i in enumerate(candidate_indices)]
    return quiet, candidates


def physical_neighbor_outcomes(sim, snapshot, probes):
    outcomes = []
    minimum = DETECTOR["min_side"]
    for index, probe in enumerate(probes):
        sim.restore(snapshot)
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in probe:
            sim.execute(action, trace)
        for _ in range(TAIL):
            sim.execute(probe[-1], trace)
        outcomes.append(bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG))
        remaining = len(probes) - index - 1
        toppled = sum(outcomes); intact = len(outcomes) - toppled
        # Screening only asks whether both sides can reach min_side.  Stop once
        # that is established, or once the missing side can no longer reach it.
        if toppled >= minimum and intact >= minimum:
            break
        if toppled + remaining < minimum or intact + remaining < minimum:
            break
    sim.restore(snapshot)
    return np.asarray(outcomes, bool)


def screen_boundaries(candidates, replay, xml, scalars, eps):
    wanted = {}
    for row in candidates:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    results = []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted[episode_id]:
                    snapshot = sim.snapshot()
                    probes = coherent_probe_chunks(
                        episode.actions[start:start + HORIZON], scalars, eps)
                    outcomes = physical_neighbor_outcomes(sim, snapshot, probes)
                    count = int(outcomes.sum()); tested = len(outcomes)
                    row = dict(wanted[episode_id][start])
                    row.update({"neighbor_toppled_probes": count,
                                "neighbor_intact_probes": int(tested - count),
                                "screened_probes": tested,
                                "supported_boundary": bool(min(count, tested - count)
                                                           >= DETECTOR["min_side"])})
                    results.append(row)
                sim.execute(action)
            print(f"  screen {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return results


def select_supported(screened, count, allow_shortfall=False):
    supported = [row for row in screened if row["supported_boundary"]]
    supported.sort(key=lambda row: (row["candidate_rank"], int(row["episode_id"]),
                                    row["chunk_start"]))
    if len(supported) < int(count) and not allow_shortfall:
        raise ValueError(f"found only {len(supported)} supported boundaries; need {count}")
    # First take one per episode, then fill by physical balance/rank.
    first, rest, seen = [], [], set()
    for row in supported:
        if row["episode_id"] not in seen:
            first.append(row); seen.add(row["episode_id"])
        else:
            rest.append(row)
    ordered = first + sorted(rest, key=lambda row: (
        abs(row["neighbor_toppled_probes"] - 25), row["candidate_rank"]))
    selected = []
    for row in ordered[:min(int(count), len(ordered))]:
        item = dict(row); item["stratum"] = "supported_neighbor_boundary"
        selected.append(item)
    return selected


def collect_latents(selected, replay, model, projection, xml, scalars, eps, device, batch_size):
    wanted = {}
    for row in selected:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    metadata, actual, predicted, physical = [], [], [], []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); actions = episode.actions
            sim.reset(int(episode_id))
            frames = deque([sim.render()], maxlen=3); props = deque([sim.proprio()], maxlen=3)
            for i in range(2):
                sim.execute(actions[i]); frames.append(sim.render()); props.append(sim.proprio())
            for start in range(2, len(actions) - HORIZON + 1, HORIZON):
                if start in wanted[episode_id]:
                    snapshot = sim.snapshot(); start_frame = frames[-1]; start_prop = props[-1]
                    probes = coherent_probe_chunks(actions[start:start + HORIZON], scalars, eps)
                    pred = model_probe_deltas(model, device, np.stack(frames), np.stack(props),
                                              actions[start - 2:start], probes, projection,
                                              batch_size)
                    real, _, diagnostics = true_probe_deltas(
                        model, device, sim, snapshot, start_frame, start_prop, probes,
                        projection, batch_size, collect_diagnostics=True)
                    neighbor = np.max(diagnostics["peak_tilt"][:, 1:], axis=1) >= TOPPLE_DEG
                    metadata.append(dict(wanted[episode_id][start])); actual.append(real)
                    predicted.append(pred); physical.append(neighbor)
                for action_index, action in enumerate(actions[start:start + HORIZON]):
                    sim.execute(action)
                    if action_index >= HORIZON - 3:
                        frames.append(sim.render()); props.append(sim.proprio())
            print(f"  encode {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return (metadata, np.stack(actual), np.stack(predicted), np.stack(physical))


def exact_metric_intervals(rows):
    boundary = [r for r in rows if r["physical_boundary"]]
    nonboundary = [r for r in rows if not r["physical_boundary"]]
    tp = sum(r["alarm"] for r in boundary); fp = sum(r["alarm"] for r in nonboundary)
    return {"recall_exact_95_percent": binomial_interval(tp, len(boundary)),
            "false_positive_rate_exact_95_percent": binomial_interval(fp, len(nonboundary))}


def paired_audit(real_rows, predicted_rows):
    def finite_correlation(key):
        x = np.asarray([row[key] for row in real_rows], float)
        y = np.asarray([row[key] for row in predicted_rows], float)
        valid = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[valid], y[valid])[0, 1]) if valid.sum() > 1 else None
    boundary = np.asarray([row["physical_boundary"] for row in real_rows], bool)
    real_ratio = np.asarray([row["adjacent_jump_ratio"] for row in real_rows], float)
    pred_ratio = np.asarray([row["adjacent_jump_ratio"] for row in predicted_rows], float)
    return {
        "alarm_agreement": float(np.mean([a["alarm"] == b["alarm"]
                                          for a, b in zip(real_rows, predicted_rows)])),
        "adjacent_jump_ratio_correlation": finite_correlation("adjacent_jump_ratio"),
        "bic_evidence_correlation": finite_correlation("jump_evidence_bic"),
        "supported_boundary_median_real_jump_ratio": float(np.median(real_ratio[boundary])),
        "supported_boundary_median_predicted_jump_ratio": float(np.median(pred_ratio[boundary])),
        "supported_boundary_predicted_to_real_jump_ratio": float(
            np.median(pred_ratio[boundary] / np.maximum(real_ratio[boundary], 1e-12))),
    }


def write_cache(path, metadata, scalars, actual, predicted, physical):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=np.asarray(json.dumps(metadata)), scalars=scalars,
                        actual_delta=actual, predicted_delta=predicted, physical=physical)


def read_cache(path):
    data = dict(np.load(path, allow_pickle=False))
    return (json.loads(str(data["metadata_json"])), data["scalars"], data["actual_delta"],
            data["predicted_delta"], data["physical"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB)); ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--scan-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--projection-cache", default=str(ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--holdout-cache", default=str(ROOT / "results/jenga/ordered_holdout_spread_cache.npz"))
    ap.add_argument("--screen-cache", default=str(ROOT / "results/jenga/neighbor_boundary_screen.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/neighbor_validation_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/neighbor_validation.json"))
    ap.add_argument("--quiet-chunks", type=int, default=250)
    ap.add_argument("--supported-boundaries", type=int, default=30)
    ap.add_argument("--boundary-candidates", type=int, default=1000)
    ap.add_argument("--probes", type=int, default=50); ap.add_argument("--eps", type=float, default=.10)
    ap.add_argument("--batch-size", type=int, default=50); ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-screen", action="store_true"); ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--allow-boundary-shortfall", action="store_true",
                    help="Evaluate every supported boundary if the fixed target is infeasible")
    args = ap.parse_args(); scalars = probe_scalars(args.probes)

    original_meta, _, _, original_actual, _ = read_probe_cache(args.original_cache)
    scale = robust_component_scale(original_actual)
    if args.reuse_cache:
        metadata, scalars, actual, predicted, physical = read_cache(args.cache)
        screened = json.loads(Path(args.screen_cache).read_text())["rows"]
    else:
        excluded = prior_keys(args.original_cache, args.holdout_cache)
        quiet, candidates = choose_pools(args.scan_cache, excluded, args.quiet_chunks,
                                         args.boundary_candidates)
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_neighbor_validation_") as temp:
                xml = extract_sim(args.sim_archive, temp)
                if args.reuse_screen:
                    screened = json.loads(Path(args.screen_cache).read_text())["rows"]
                    completed = {(row["episode_id"], row["chunk_start"]) for row in screened}
                    remaining = [row for row in candidates
                                 if (row["episode_id"], row["chunk_start"]) not in completed]
                    if remaining:
                        screened.extend(screen_boundaries(remaining, replay, xml, scalars,
                                                          args.eps))
                        screened.sort(key=lambda row: row["candidate_rank"])
                        Path(args.screen_cache).write_text(json.dumps(
                            {"rows": screened}, indent=2) + "\n")
                else:
                    screened = screen_boundaries(candidates, replay, xml, scalars, args.eps)
                    Path(args.screen_cache).write_text(json.dumps({"rows": screened}, indent=2) + "\n")
                boundaries = select_supported(screened, args.supported_boundaries,
                                               args.allow_boundary_shortfall)
                model = load_world_model(args.checkpoint, args.device)
                metadata, actual, predicted, physical = collect_latents(
                    quiet + boundaries, replay, model, projection_basis(args.projection_cache),
                    xml, scalars, args.eps, args.device, args.batch_size)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, actual, predicted, physical)

    real_rows = score_responses(scalars, actual, physical, metadata, scale=scale, **DETECTOR)
    predicted_rows = score_responses(scalars, predicted, physical, metadata, scale=scale, **DETECTOR)
    real_summary = summarise(real_rows); predicted_summary = summarise(predicted_rows)
    real_summary.update(exact_metric_intervals(real_rows)); predicted_summary.update(exact_metric_intervals(predicted_rows))
    real_summary["episode_bootstrap_95_percent"] = bootstrap_confidence(real_rows)
    predicted_summary["episode_bootstrap_95_percent"] = bootstrap_confidence(predicted_rows)
    controls = [r for r in real_rows if r["stratum"] == "quiet_control"]
    benchmark = [r for r in real_rows if r["stratum"] == "supported_neighbor_boundary"]
    gates = {"benchmark_size_ge_30": bool(len(benchmark) >= args.supported_boundaries),
             "quiet_control_fpr_le_0_05": bool(
                 sum(r["alarm"] for r in controls) / max(len(controls), 1) <= .05),
             "supported_boundary_recall_ge_0_80": bool(
                 sum(r["alarm"] for r in benchmark) / max(len(benchmark), 1) >= .80)}
    result = {
        "protocol": {"horizon": HORIZON, "held_tail": TAIL, "eps": args.eps,
                     "probes": len(scalars), "frozen_detector": DETECTOR,
                     "component_scale_source": f"original {len(original_meta)}-chunk real-delta set",
                     "quiet_selection": "temporally spread nominally quiet chunks, excluding previously probed states",
                     "boundary_selection": "nominal-near-45 pool physically screened; only >=8/side neighbor boundaries retained",
                     "boundary_benchmark_is_prevalence_sample": False},
        "screen": {"candidate_chunks": len(screened),
                   "supported_boundaries_found": int(sum(r["supported_boundary"] for r in screened))},
        "real_encoded": real_summary, "predicted_world_model": predicted_summary,
        "paired_real_prediction_audit": paired_audit(real_rows, predicted_rows),
        "acceptance_gates": gates, "passed": bool(all(gates.values())),
        "real_rows": real_rows, "predicted_rows": predicted_rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_rows")}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
