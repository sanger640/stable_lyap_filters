"""Nominal-centered, short-tail persistent branch margins on physical Jenga states.

Unlabeled candidate: different known physical basin at both held tails 5 and 10,
supported at two consecutive strengths. The existing 45-degree endpoint topple
criterion is a separate diagnostic oracle, never used to fit or select basins.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basins import fit_basin_model  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses, physical_state  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_physical_basin_tail_compare import load_cache  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_trajectory_oracle import read_cache  # noqa: E402

TAILS = (5, 10)
MAGNITUDES = np.asarray([0, .01, .02, .03, .04, .06, .08, .10, .12,
                         .15, .20, .25, .30, .40, .50, .60], np.float32)
COEFFICIENTS = np.concatenate([-MAGNITUDES[:0:-1], MAGNITUDES])


def spaced_indices(indices, count):
    if not indices:
        return []
    return [indices[i] for i in np.unique(np.linspace(0, len(indices) - 1,
                                                     min(count, len(indices)), dtype=int))]


def choose_states(boundary_cache, expanded_cache, contact_result):
    boundary_meta, _, _, _ = read_cache(boundary_cache)
    boundaries = [dict(r) for r in boundary_meta
                  if r["stratum"] == "recentered_neighbor_boundary"]
    # Split by episode, with the later 15 never entering the fitted atlas.
    boundaries.sort(key=lambda r: int(r["episode_id"]))
    heldout = boundaries[len(boundaries) // 2:]
    train = boundaries[:len(boundaries) // 2]
    expanded_meta, _, _, outcomes = read_cache(expanded_cache)
    moving = [dict(expanded_meta[i]) for i in range(len(expanded_meta))
              if expanded_meta[i]["stratum"] == "moving_nominal_non_topple"
              and not outcomes[i].any()]
    silent = [dict(expanded_meta[i]) for i in range(len(expanded_meta))
              if expanded_meta[i]["stratum"] == "silent_episode" and not outcomes[i].any()]
    contact = [dict(r) for r in json.loads(Path(contact_result).read_text())["rows"]
               if r["stratum"] == "nominal_contact_lift"
               and r["physical_counts"]["toppled"] == 0]
    pools = [(heldout, len(heldout)), (moving, 20), (silent, 10), (contact, 10)]
    chosen, seen = [], set()
    for pool, count in pools:
        for row in spaced_indices(list(range(len(pool))), count):
            item = pool[row]
            key = (item["episode_id"], item["chunk_start"])
            if key not in seen:
                chosen.append({"episode_id": item["episode_id"],
                               "chunk_start": item["chunk_start"],
                               "stratum": item["stratum"]})
                seen.add(key)
    return train, chosen


def simulate_grid(sim, snapshot, chunk, coefficients):
    probes = offset_probe_chunks(chunk, coefficients, 1.0)
    start = physical_state(sim)
    endpoint = np.empty((len(coefficients), len(TAILS), len(start)), np.float32)
    tilt = np.empty((len(coefficients), len(TAILS)), np.float32)
    for i, probe in enumerate(probes):
        sim.restore(snapshot)
        for action in probe:
            sim.execute(action)
        for held in range(1, max(TAILS) + 1):
            sim.execute(probe[-1])
            if held in TAILS:
                j = TAILS.index(held)
                endpoint[i, j] = physical_state(sim) - start
                tilt[i, j] = np.max(sim.block_diagnostics()["tilt"][1:])
    sim.restore(snapshot)
    return endpoint, tilt


def collect(rows, replay, xml, coefficients):
    wanted = {(r["episode_id"], r["chunk_start"]): r for r in rows}
    episode_ids = sorted({r["episode_id"] for r in rows}, key=int)
    sim = DirectJengaSim(xml)
    metadata, endpoints, tilts, span_mm = [], [], [], []
    try:
        for number, episode_id in enumerate(episode_ids):
            episode = replay.episode(episode_id)
            sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                key = (episode_id, start)
                if key in wanted:
                    chunk = episode.actions[start:start + HORIZON]
                    values, angles = simulate_grid(sim, sim.snapshot(), chunk, coefficients)
                    metadata.append(wanted[key]); endpoints.append(values); tilts.append(angles)
                    span_mm.append(float(1000 * np.max(np.linalg.norm(
                        chunk[:, :3] - chunk[0, :3], axis=1))))
                sim.execute(action)
            print(f"  margin replay {number + 1}/{len(episode_ids)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return metadata, np.stack(endpoints), np.stack(tilts), np.asarray(span_mm)


def fit_atlas(original_cache, tail_cache, train_boundary, excluded_episodes):
    metadata, _, endpoints, _ = load_cache(tail_cache)
    keys = {(r["episode_id"], r["chunk_start"]) for r in train_boundary
            if r["episode_id"] not in excluded_episodes}
    boundary = np.asarray([(r["episode_id"], r["chunk_start"]) in keys for r in metadata])
    source = np.load(original_cache, allow_pickle=False)
    original_meta = json.loads(str(source["metadata_json"]))
    keep = np.asarray([r["episode_id"] not in excluded_episodes for r in original_meta])
    original = original_physical_responses(original_cache)[keep]
    models, scales = [], []
    for j in range(len(TAILS)):
        reference = np.concatenate([original, endpoints[boundary, :, j]], axis=0)
        scale = robust_component_scale(reference)
        model = fit_basin_model((reference / scale).reshape(-1, reference.shape[-1]),
                                pca_dim=4, min_cluster_fraction=.10)
        models.append(model); scales.append(scale)
    return models, scales, int(keep.sum()), int(boundary.sum())


def nearest_branch(coefficients, signatures):
    """First sustained regime departure on either side of nominal; noise abstains.

    signatures is (n_strength, n_tail), with -1 for unknown. Each tail may use
    a different label vocabulary; comparisons are to that tail's nominal label.
    """
    coeff = np.asarray(coefficients)
    sig = np.asarray(signatures)
    zero = int(np.flatnonzero(np.isclose(coeff, 0))[0])
    base = sig[zero]
    if np.any(base < 0):
        return {"status": "nominal_unassigned", "nominal_labels": base.tolist()}
    candidates = []
    for direction in (-1, 1):
        path = list(range(zero, -1 if direction < 0 else len(coeff), direction))
        for p in range(1, len(path) - 1):
            i, farther = path[p], path[p + 1]
            if np.all(sig[i] >= 0) and np.all(sig[i] != base) and np.array_equal(sig[i], sig[farther]):
                last_same = max((q for q in range(0, p) if np.array_equal(sig[path[q]], base)),
                                default=0)
                gap = bool(np.any(sig[path[last_same + 1:p]] < 0))
                candidates.append({"direction": direction,
                                   "lower_abs_coefficient": float(abs(coeff[path[last_same]])),
                                   "upper_abs_coefficient": float(abs(coeff[i])),
                                   "support_gap": gap,
                                   "branch_labels": sig[i].tolist()})
                break
    if not candidates:
        edges = []
        for i in (0, len(coeff) - 1):
            if np.all(sig[i] >= 0) and np.all(sig[i] != base):
                edges.append(float(abs(coeff[i])))
        if edges:
            return {"status": "edge_unconfirmed", "nominal_labels": base.tolist(),
                    "candidate_abs_coefficient": min(edges)}
        return {"status": "not_found_within_grid", "nominal_labels": base.tolist(),
                "tested_abs_coefficient": float(np.max(np.abs(coeff)))}
    best = min(candidates, key=lambda r: r["upper_abs_coefficient"])
    return {"status": "bracketed", "nominal_labels": base.tolist(), **best}


def summarise(rows):
    strata = sorted({r["stratum"] for r in rows})
    result = {}
    for stratum in strata:
        part = [r for r in rows if r["stratum"] == stratum]
        basin_found = [r["basin_margin"] for r in part
                       if r["basin_margin"]["status"] == "bracketed"]
        oracle_found = [r["oracle_margin"] for r in part
                        if r["oracle_margin"]["status"] == "bracketed"]
        paired = [r for r in part if r["basin_margin"]["status"] == "bracketed"
                  and r["oracle_margin"]["status"] == "bracketed"]
        result[stratum] = {"states": len(part),
                           "prior_trajectory_alarm_cases": int(sum(
                               r["prior_trajectory_alarm"] is not None for r in part)),
                           "prior_trajectory_alarms": int(sum(
                               r["prior_trajectory_alarm"] is True for r in part)),
                           "basin_bracketed": len(basin_found),
                           "basin_nominal_unassigned": int(sum(
                               r["basin_margin"]["status"] == "nominal_unassigned" for r in part)),
                           "oracle_bracketed": len(oracle_found),
                           "oracle_edge_unconfirmed": int(sum(
                               r["oracle_margin"]["status"] == "edge_unconfirmed" for r in part)),
                           "basin_and_oracle_bracketed": len(paired),
                           "basin_upper_mm_median_when_found": (float(np.median([
                               m["upper_mm"] for m in basin_found])) if basin_found else None),
                           "oracle_upper_mm_median_when_found": (float(np.median([
                               m["upper_mm"] for m in oracle_found])) if oracle_found else None),
                           "mean_basin_grid_coverage": float(np.mean([
                               r["basin_known_fraction"] for r in part]))}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--tail-cache", default=str(ROOT / "results/jenga/physical_basin_tails_cache.npz"))
    ap.add_argument("--boundary-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--expanded-cache", default=str(ROOT / "results/jenga/expanded_controls_cache.npz"))
    ap.add_argument("--expanded-result", default=str(ROOT / "results/jenga/expanded_controls.json"))
    ap.add_argument("--contact-result", default=str(ROOT / "results/jenga/contact_pick_check.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/persistent_branch_margin_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    train, test = choose_states(args.boundary_cache, args.expanded_cache, args.contact_result)
    excluded = {r["episode_id"] for r in test}
    models, scales, original_count, boundary_count = fit_atlas(
        args.original_cache, args.tail_cache, train, excluded)
    if args.reuse_cache:
        data = np.load(args.cache, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        endpoints, tilts, span_mm, coefficients = (data[k] for k in
                                                   ("endpoints", "tilts", "span_mm", "coefficients"))
    else:
        coefficients = COEFFICIENTS
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_branch_margin_") as temp:
                metadata, endpoints, tilts, span_mm = collect(
                    test, replay, extract_sim(args.sim_archive, temp), coefficients)
        finally:
            replay.close()
        target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, metadata_json=np.asarray(json.dumps(metadata)),
                            coefficients=coefficients, endpoints=endpoints, tilts=tilts,
                            span_mm=span_mm)
    labels = np.empty((len(metadata), len(coefficients), len(TAILS)), int)
    prior = {(r["episode_id"], r["chunk_start"]): r["alarm"] for r in
             json.loads(Path(args.expanded_result).read_text())["rows"]}
    for j, (model, scale) in enumerate(zip(models, scales)):
        labels[:, :, j] = model.predict(
            (endpoints[:, :, j] / scale).reshape(-1, endpoints.shape[-1])).reshape(
                len(metadata), len(coefficients))
    rows = []
    for meta, lab, tilt, span in zip(metadata, labels, tilts, span_mm):
        basin = nearest_branch(coefficients, lab)
        oracle = nearest_branch(coefficients, (tilt >= TOPPLE_DEG).astype(int))
        for margin in (basin, oracle):
            if margin["status"] == "bracketed":
                margin["lower_mm"] = margin["lower_abs_coefficient"] * float(span)
                margin["upper_mm"] = margin["upper_abs_coefficient"] * float(span)
            elif margin["status"] == "not_found_within_grid":
                margin["tested_mm"] = margin["tested_abs_coefficient"] * float(span)
            elif margin["status"] == "edge_unconfirmed":
                margin["candidate_mm"] = margin["candidate_abs_coefficient"] * float(span)
        rows.append({**meta, "chunk_span_mm": float(span), "basin_margin": basin,
                     "oracle_margin": oracle,
                     "prior_trajectory_alarm": prior.get(
                         (meta["episode_id"], meta["chunk_start"]), None),
                     "basin_known_fraction": float((lab >= 0).mean()),
                     "nominal_neighbor_tilt_deg": tilt[len(coefficients) // 2].tolist()})
    result = {"protocol": {"nominal_centered": True, "horizon": HORIZON,
                            "held_tails": list(TAILS), "coefficients": coefficients.tolist(),
                            "persistent": "different from nominal at both tails and same "
                                          "alternative labels on two consecutive strengths",
                            "basin": "two unlabeled tail-specific PCA4/HDBSCAN atlases; noise abstains",
                            "oracle": "endpoint neighbor tilt >=45 degrees at both tails, "
                                      "diagnostic only; not used to fit basins",
                            "reference_original_states": original_count,
                            "reference_boundary_states": boundary_count,
                            "atlas_clusters_by_tail": {str(t): m.n_clusters
                                                       for t, m in zip(TAILS, models)},
                            "atlas_reference_coverage_by_tail": {str(t): m.coverage
                                                                for t, m in zip(TAILS, models)},
                            "test_episodes_excluded_from_reference": True,
                            "cohort_selection": "held-out constructed boundaries and controls "
                                                "preselected using physical outcomes; evaluation "
                                                "only, not detector training",
                            "strength_note": "broad diagnostic grid, not validated controller error range"},
              "summary": summarise(rows), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
