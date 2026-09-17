"""Pilot: equal probe counts at physically smaller action perturbations."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from basins import dissent_count, fit_basin_model, known_coverage  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_short_held_tails import extract_sim  # noqa: E402
from jenga_physical_basin_tail_compare import load_cache  # noqa: E402
from jenga_trajectory_oracle import collect, read_cache, score_rows  # noqa: E402


def sample_rows(metadata, outcomes, stratum, count, safe_only=False):
    indices = [i for i, row in enumerate(metadata) if row["stratum"] == stratum
               and (not safe_only or not outcomes[i].any())]
    chosen = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [indices[i] for i in chosen]


def fit_frozen_basins(original_cache, tail_cache):
    metadata, _, endpoints, _ = load_cache(tail_cache)
    boundary = np.asarray([r["stratum"] == "recentered_neighbor_boundary"
                           and r["episode_id"] not in ("24", "74") for r in metadata])
    original = original_physical_responses(original_cache)
    reference = np.concatenate([original, endpoints[boundary, :, 0]], axis=0)
    scale = robust_component_scale(reference)
    model = fit_basin_model((reference / scale).reshape(-1, reference.shape[-1]),
                            pca_dim=4, min_cluster_fraction=.10)
    return model, scale


def basin_rows(metadata, trajectories, model, scale):
    endpoints = trajectories[:, :, -1]
    labels = model.predict((endpoints / scale).reshape(-1, endpoints.shape[-1]))
    labels = labels.reshape(len(metadata), endpoints.shape[1])
    return [{"episode_id": meta["episode_id"], "chunk_start": meta["chunk_start"],
             "stratum": meta["stratum"], "coverage": known_coverage(label, model.n_clusters),
             "dissent": dissent_count(label, model.n_clusters),
             "alarm": bool(dissent_count(label, model.n_clusters) >= 2)}
            for meta, label in zip(metadata, labels)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--benchmark-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    parser.add_argument("--expanded-cache", default=str(ROOT / "results/jenga/expanded_controls_cache.npz"))
    parser.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    parser.add_argument("--tail-cache", default=str(ROOT / "results/jenga/physical_basin_tails_cache.npz"))
    parser.add_argument("--output", default=str(ROOT / "results/jenga/action_strength_resim.json"))
    args = parser.parse_args()
    bench = read_cache(args.benchmark_cache)
    expanded = read_cache(args.expanded_cache)
    scalars = bench[1]
    if not np.allclose(scalars, expanded[1]):
        raise ValueError("different scalar grids")
    boundary_indices = sample_rows(bench[0], bench[3], "recentered_neighbor_boundary", 10)
    moving_indices = sample_rows(expanded[0], expanded[3], "moving_nominal_non_topple", 10,
                                 safe_only=True)
    metadata = [dict(bench[0][i]) for i in boundary_indices]
    metadata += [dict(expanded[0][i]) for i in moving_indices]
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    basin_model, basin_scale = fit_frozen_basins(args.original_cache, args.tail_cache)
    original_trajectories = np.concatenate([bench[2][boundary_indices],
                                            expanded[2][moving_indices]])
    original_outcomes = np.concatenate([bench[3][boundary_indices],
                                        expanded[3][moving_indices]])
    results = {}
    original_scores = score_rows(scalars, original_trajectories,
                                 original_outcomes, metadata, scale)
    results["0.1"] = original_scores
    basin_results = {"0.1": basin_rows(metadata, original_trajectories,
                                         basin_model, basin_scale)}
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_strength_pilot_") as temp:
            xml = extract_sim(args.sim_archive, temp)
            for eps in (.035, .0165):
                rows, trajectories, outcomes = collect(metadata, replay, xml, scalars, eps)
                results[str(eps)] = score_rows(scalars, trajectories, outcomes, rows, scale)
                basin_results[str(eps)] = basin_rows(rows, trajectories,
                                                      basin_model, basin_scale)
    finally:
        replay.close()
    compact = {}
    for eps, rows in results.items():
        boundary = [r for r in rows if r["stratum"] == "recentered_neighbor_boundary"]
        moving = [r for r in rows if r["stratum"] == "moving_nominal_non_topple"]
        basin = basin_results[eps]
        basin_boundary = [r for r in basin if r["stratum"] == "recentered_neighbor_boundary"]
        basin_moving = [r for r in basin if r["stratum"] == "moving_nominal_non_topple"]
        compact[eps] = {
            "max_abs_perturbation_coefficient": float(float(eps) * np.max(np.abs(scalars))),
            "boundary_alarms": int(sum(r["alarm"] for r in boundary)),
            "boundary_with_topple_branch": int(sum(0 < r["physical_counts"]["toppled"] < 50
                                                  for r in boundary)),
            "moving_safe_alarms": int(sum(r["alarm"] for r in moving
                                          if r["physical_counts"]["toppled"] == 0)),
            "moving_still_safe": int(sum(r["physical_counts"]["toppled"] == 0
                                         for r in moving)),
            "moving_with_topple": int(sum(r["physical_counts"]["toppled"] > 0
                                           for r in moving)),
            "basin_boundary_alarms": int(sum(r["alarm"] for r in basin_boundary)),
            "basin_boundary_mean_coverage": float(np.mean([r["coverage"]
                                                              for r in basin_boundary])),
            "basin_moving_alarms": int(sum(r["alarm"] for r in basin_moving)),
            "basin_moving_mean_coverage": float(np.mean([r["coverage"]
                                                            for r in basin_moving])),
            "case_rows": [{"episode_id": r["episode_id"],
                           "chunk_start": r["chunk_start"],
                           "stratum": r["stratum"],
                           "alarm": r["alarm"],
                           "score": r["trajectory_score"],
                           "toppled_probes": r["physical_counts"]["toppled"]}
                          for r in rows]}
    result = {"protocol": {"pilot": True, "selection": "10 evenly spaced recentered boundary "
                            "families and 10 evenly spaced moving-safe states; same 50 probe "
                            "scalars per strength", "horizon": 8, "held_tail": 5,
                            "frozen_scale_and_BIC": True,
                            "frozen_basin": "original diverse endpoints plus tail-5 boundary "
                                            "endpoints; PCA4 HDBSCAN, noise excluded; selected "
                                            "boundary families overlap atlas, so their detection "
                                            "is optimistic",
                            "center_offset_note": "original recenter offsets retained at every strength"},
              "results": compact}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {j: v for j, v in section.items() if j != "case_rows"}
                      for k, section in compact.items()}, indent=2))


if __name__ == "__main__":
    main()
