"""Episode-held-out validation of the frozen 4x ordered jump detector."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay, load_world_model  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import (coherent_probe_chunks, probe_scalars,
                                      read_probe_cache, true_probe_deltas)  # noqa: E402
from jenga_ordered_change_points import score_responses, summarise  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim,
                                    projection_basis)  # noqa: E402


def heldout_quiet_chunks(short_cache, original_probe_cache, count):
    """Choose nominally quiet chunks only from episodes unseen by local probing."""
    original, *_ = read_probe_cache(original_probe_cache)
    used_episodes = {row["episode_id"] for row in original}
    data = dict(np.load(short_cache, allow_pickle=False))
    ids = data["episode_ids"].astype(str)
    starts = data["chunk_starts"].astype(int)
    peak5 = np.asarray(data["tail5_peak_tilt"], np.float32)
    end0 = np.asarray(data["tail0_end_tilt"], np.float32)
    start_tilt = np.zeros(len(ids), np.float32)
    previous_episode, previous_tilt = None, 0.0
    for i, episode in enumerate(ids):
        if episode != previous_episode:
            previous_tilt = 0.0
        start_tilt[i] = previous_tilt
        previous_tilt = float(end0[i]); previous_episode = episode
    by_episode = {}
    for i in range(len(ids)):
        if ids[i] not in used_episodes and start_tilt[i] < TOPPLE_DEG and peak5[i] < 5.0:
            by_episode.setdefault(ids[i], []).append(i)
    episodes = sorted(by_episode, key=int)
    selected = []
    # For the default 50/25 design, choose two interior temporal quantiles per
    # episode.  Taking rank 0/1 would test only the easy pre-interaction prefix.
    per_episode = max(1, int(np.ceil(int(count) / max(len(episodes), 1))))
    for episode in episodes:
        candidates = by_episode[episode]
        positions = np.linspace(0, len(candidates) - 1, per_episode + 2)[1:-1]
        for position in np.unique(np.round(positions).astype(int)):
            selected.append(candidates[position])
    selected = selected[:int(count)]
    if len(selected) < int(count):
        raise ValueError(f"only {len(selected)} held-out quiet chunks are available")
    return [{"episode_id": ids[i], "chunk_start": int(starts[i]),
             "stratum": "quiet_control", "heldout_episode": True,
             "screen_start_tilt": float(start_tilt[i]),
             "screen_peak_tail5": float(peak5[i])} for i in selected]


def collect(args, selected, replay, model, projection, xml):
    by_episode = {}
    for row in selected:
        by_episode.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    scalars = probe_scalars(args.probes); rows = []; actual = []; physical = []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(by_episode, key=int)):
            episode = replay.episode(episode_id); wanted = by_episode[episode_id]
            sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted:
                    snapshot = sim.snapshot(); start_frame = sim.render(); start_prop = sim.proprio()
                    probes = coherent_probe_chunks(
                        episode.actions[start:start + HORIZON], scalars, args.eps)
                    delta, peak, _ = true_probe_deltas(
                        model, args.device, sim, snapshot, start_frame, start_prop, probes,
                        projection, args.batch_size, collect_diagnostics=False)
                    meta = dict(wanted[start]); meta["start_tilt"] = float(sim.max_tilt())
                    rows.append(meta); actual.append(delta); physical.append(peak >= TOPPLE_DEG)
                sim.execute(action)
            print(f"  {number + 1}/{len(by_episode)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return rows, scalars, np.stack(actual), np.stack(physical)


def write_cache(path, rows, scalars, actual, physical):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=np.asarray(json.dumps(rows)),
                        scalars=np.asarray(scalars, np.float32),
                        actual_delta=np.asarray(actual, np.float32),
                        physical=np.asarray(physical, bool))


def read_cache(path):
    data = dict(np.load(path, allow_pickle=False))
    return (json.loads(str(data["metadata_json"])), data["scalars"],
            data["actual_delta"], data["physical"])


def binomial_interval(successes, trials, alpha=0.05):
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--short-cache", default=str(ROOT / "results/jenga/short_held_tails_cache.npz"))
    ap.add_argument("--original-cache", default=str(
        ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--projection-cache", default=str(
        ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--cache", default=str(
        ROOT / "results/jenga/ordered_holdout_spread_cache.npz"))
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/ordered_holdout.json"))
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--probes", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    # These settings are intentionally frozen before reading holdout responses.
    detector = {"dimension": 4, "degree": 3, "min_side": 8, "min_jump_ratio": 4.0}
    original_metadata, _, _, original_actual, _ = read_probe_cache(args.original_cache)
    original_scale = robust_component_scale(original_actual)
    if args.reuse_cache:
        metadata, scalars, actual, physical = read_cache(args.cache)
    else:
        selected = heldout_quiet_chunks(args.short_cache, args.original_cache, args.chunks)
        replay = JengaReplay(args.lmdb)
        model = load_world_model(args.checkpoint, args.device)
        projection = projection_basis(args.projection_cache)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_ordered_holdout_") as temp:
                metadata, scalars, actual, physical = collect(
                    args, selected, replay, model, projection,
                    extract_sim(args.sim_archive, temp))
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, actual, physical)
    rows = score_responses(
        scalars, actual, physical, metadata, scale=original_scale, **detector)
    summary = summarise(rows)
    nonboundaries = [row for row in rows if not row["physical_boundary"]]
    false_alarms = sum(row["alarm"] for row in nonboundaries)
    gate_fpr = false_alarms / max(len(nonboundaries), 1)
    result = {
        "protocol": {"selection": "two temporally spread nominally quiet chunks from each episode absent from original local-probe experiment",
                     "chunks": len(rows), "episodes": len(set(row["episode_id"] for row in rows)),
                     "probes": len(scalars), "horizon": 8, "held_tail": 5,
                     "frozen_detector": detector,
                     "component_scale_source": f"original {len(original_metadata)}-chunk real-delta set"},
        "summary": summary,
        "physical_boundary_chunks": int(sum(row["physical_boundary"] for row in rows)),
        "nonboundary_false_alarms": int(false_alarms),
        "nonboundary_chunks": len(nonboundaries),
        "heldout_false_positive_rate": gate_fpr,
        "heldout_false_positive_rate_exact_95_percent": binomial_interval(
            false_alarms, len(nonboundaries)),
        "real_control_gate": {"maximum_fpr": 0.05, "passed": bool(gate_fpr <= 0.05)},
        "rows": rows,
        "note": "No detector setting or component scaling was estimated from holdout responses.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("ordered heldout result")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
