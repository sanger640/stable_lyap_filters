"""Unlabeled physical-regime dissent at 5/10/20 held steps, with full-episode audit."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basins import dissent_count, fit_basin_model, known_coverage  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses, physical_state  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_trajectory_oracle import read_cache  # noqa: E402

TAILS = (5, 10, 20)


def select_states(boundary_cache, episode_cache, episode_ids):
    boundary, scalars, _, _ = read_cache(boundary_cache)
    full, full_scalars, _, _ = read_cache(episode_cache)
    if not np.allclose(scalars, full_scalars):
        raise ValueError("probe scalars differ between caches")
    selected = [dict(row) for row in boundary
                if row["stratum"] == "recentered_neighbor_boundary"]
    selected += [dict(row) for row in full if row["episode_id"] in episode_ids]
    return selected, scalars


def simulate_tails(sim, snapshot, probes):
    start = physical_state(sim)
    endpoints = np.empty((len(probes), len(TAILS), len(start)), np.float32)
    outcomes = np.empty((len(probes), len(TAILS)), bool)
    for i, probe in enumerate(probes):
        sim.restore(snapshot)
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in probe:
            sim.execute(action, trace)
        for held in range(1, max(TAILS) + 1):
            sim.execute(probe[-1], trace)
            if held in TAILS:
                j = TAILS.index(held)
                endpoints[i, j] = physical_state(sim) - start
                outcomes[i, j] = bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG)
    sim.restore(snapshot)
    return endpoints, outcomes


def collect(states, replay, xml, scalars):
    wanted = {}
    for row in states:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    metadata, endpoints, outcomes = [], [], []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id)
            sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted[episode_id]:
                    row = wanted[episode_id][start]
                    probes = offset_probe_chunks(
                        episode.actions[start:start + HORIZON], scalars, .10,
                        row.get("center_offset", 0.0))
                    values, truth = simulate_tails(sim, sim.snapshot(), probes)
                    metadata.append(row); endpoints.append(values); outcomes.append(truth)
                sim.execute(action)
            print(f"  basin tails {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return metadata, np.stack(endpoints), np.stack(outcomes)


def save_cache(path, metadata, scalars, endpoints, outcomes):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, metadata_json=np.asarray(json.dumps(metadata)),
                        scalars=scalars, endpoints=endpoints, outcomes=outcomes)


def load_cache(path):
    data = np.load(path, allow_pickle=False)
    return (json.loads(str(data["metadata_json"])), data["scalars"],
            data["endpoints"], data["outcomes"])


def score_tail(tail_index, metadata, endpoints, outcomes, original_cache, excluded_episodes):
    original = np.load(original_cache, allow_pickle=False)
    source_metadata = json.loads(str(original["metadata_json"]))
    original_ends = original_physical_responses(original_cache)
    keep = np.asarray([r["episode_id"] not in excluded_episodes for r in source_metadata])
    boundary = np.asarray([r["stratum"] == "recentered_neighbor_boundary"
                           and r["episode_id"] not in excluded_episodes for r in metadata])
    reference = np.concatenate([original_ends[keep], endpoints[boundary, :, tail_index]], axis=0)
    scale = robust_component_scale(reference)
    model = fit_basin_model((reference / scale).reshape(-1, reference.shape[-1]),
                            pca_dim=4, min_cluster_fraction=.10)
    labels = model.predict((endpoints[:, :, tail_index] / scale).reshape(-1, endpoints.shape[-1]))
    labels = labels.reshape(len(metadata), len(outcomes[0]))
    rows = []
    for meta, label, truth in zip(metadata, labels, outcomes[:, :, tail_index]):
        counts = [int(np.sum(label == c)) for c in range(model.n_clusters)]
        k = dissent_count(label, model.n_clusters)
        rows.append({"episode_id": meta["episode_id"], "chunk_start": meta["chunk_start"],
                     "stratum": meta["stratum"], "known_counts": counts,
                     "noise_count": int(np.sum(label < 0)),
                     "coverage": known_coverage(label, model.n_clusters),
                     "dissent": k, "alarm": bool(k >= 2),
                     "physical_toppled_probes": int(truth.sum())})
    truth_reference = np.concatenate([original["physical"][keep],
                                      outcomes[boundary, :, tail_index]], axis=0).reshape(-1)
    reference_composition = []
    for c in range(model.n_clusters):
        member = model.labels == c
        reference_composition.append({"size": int(member.sum()),
                                      "toppled_fraction_diagnostic_only": float(
                                          truth_reference[member].mean())})
    boundary_rows = [r for r in rows if r["stratum"] == "recentered_neighbor_boundary"]
    episode_rows = [r for r in rows if r["stratum"] != "recentered_neighbor_boundary"]
    summary = {"reference_states": int(len(reference)),
               "reference_probe_endpoints": int(reference.shape[0] * reference.shape[1]),
               "discovered_clusters": model.n_clusters,
               "reference_coverage": model.coverage,
               "reference_cluster_composition": reference_composition,
               "boundary_detections": int(sum(r["alarm"] for r in boundary_rows)),
               "boundary_cases": len(boundary_rows),
               "boundary_median_coverage": float(np.median([r["coverage"] for r in boundary_rows])),
               "boundary_mean_coverage": float(np.mean([r["coverage"] for r in boundary_rows])),
               "episode_alarm_counts": {ep: int(sum(r["alarm"] for r in episode_rows
                                                    if r["episode_id"] == ep))
                                        for ep in sorted(excluded_episodes, key=int)},
               "episode_mean_coverage": {ep: float(np.mean([r["coverage"] for r in episode_rows
                                                             if r["episode_id"] == ep]))
                                         for ep in sorted(excluded_episodes, key=int)}}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=["24", "74"])
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--boundary-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--episode-cache", default=str(ROOT / "results/jenga/full_episode_alarm_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/physical_basin_tails_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/physical_basin_tails.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    if args.reuse_cache:
        metadata, scalars, endpoints, outcomes = load_cache(args.cache)
    else:
        selected, scalars = select_states(args.boundary_cache, args.episode_cache,
                                           set(args.episodes))
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_basin_tails_") as temp:
                metadata, endpoints, outcomes = collect(
                    selected, replay, extract_sim(args.sim_archive, temp), scalars)
        finally:
            replay.close()
        save_cache(args.cache, metadata, scalars, endpoints, outcomes)
    summaries, rows_by_tail = {}, {}
    for j, tail in enumerate(TAILS):
        summaries[str(tail)], rows_by_tail[str(tail)] = score_tail(
            j, metadata, endpoints, outcomes, args.original_cache, set(args.episodes))
    result = {"protocol": {"horizon": HORIZON, "held_tails": list(TAILS),
                            "probes": len(scalars), "eps": .10,
                            "state": "neighbor position, orientation, neighbor-involving contacts",
                            "reference": "original diverse 100-probe cache plus 30 boundary-family endpoints; "
                                         "no outcome labels used for fitting; boundary sampling is outcome-enriched",
                            "reference_excludes_test_episodes": list(args.episodes),
                            "basin": "PCA4 + HDBSCAN min_cluster_fraction=.10; nearest-member 99th percentile support",
                            "alarm": "known-basin dissent >=2, noise excluded and coverage reported",
                            "reference_overlap_warning": "boundary detection is in-sample for most boundary families; "
                                                         "full episodes are held out by episode"},
              "summary": summaries, "rows": rows_by_tail}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
