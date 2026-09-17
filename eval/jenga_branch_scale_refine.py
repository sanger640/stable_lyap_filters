"""Adaptive two-level probe refinement of frozen local branch candidates."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from branch_scale import choose_larger_half, compare_scales  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_persistent_branch import fit_unlabeled_projection  # noqa: E402
from jenga_multipeak_oracle import physical_state  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402

TAILS = (5, 10)


def simulate_coefficient(sim, snapshot, chunk, coefficient):
    sim.restore(snapshot)
    start = physical_state(sim)
    probe = offset_probe_chunks(chunk, np.asarray([coefficient], np.float32), 1.0)[0]
    result = np.empty((len(TAILS), len(start)), np.float32)
    for action in probe:
        sim.execute(action)
    for held in range(1, max(TAILS) + 1):
        sim.execute(probe[-1])
        if held in TAILS:
            result[TAILS.index(held)] = physical_state(sim) - start
    sim.restore(snapshot)
    return result


def refine_one(sim, snapshot, chunk, left_coeff, right_coeff, left_response,
               right_response, project):
    left_replay = simulate_coefficient(sim, snapshot, chunk, left_coeff)
    right_replay = simulate_coefficient(sim, snapshot, chunk, right_coeff)
    replay_error = max(float(np.linalg.norm(left_replay - left_response)),
                       float(np.linalg.norm(right_replay - right_response)))
    replay_error_pose = max(float(np.linalg.norm(left_replay[:,:18] - left_response[:,:18])),
                            float(np.linalg.norm(right_replay[:,:18] - right_response[:,:18])))
    replay_error_contact = max(float(np.linalg.norm(left_replay[:,18:] - left_response[:,18:])),
                               float(np.linalg.norm(right_replay[:,18:] - right_response[:,18:])))
    left = project(left_replay)
    right = project(right_replay)
    distances = [np.linalg.norm(right - left, axis=1)]
    coefficients, new_responses, interval_levels = [], [], []
    a, b = float(left_coeff), float(right_coeff)
    for _ in range(2):
        midpoint_coeff = (a + b) / 2
        midpoint_response = simulate_coefficient(sim, snapshot, chunk, midpoint_coeff)
        midpoint = project(midpoint_response)
        half = choose_larger_half(left, midpoint, right)
        if half == 0:
            b, right = midpoint_coeff, midpoint
        else:
            a, left = midpoint_coeff, midpoint
        coefficients.append(midpoint_coeff)
        new_responses.append(midpoint_response)
        interval_levels.append([a, b])
        distances.append(np.linalg.norm(right - left, axis=1))
    return (np.stack(distances, axis=1), np.asarray(coefficients, np.float32),
            np.stack(new_responses), interval_levels, replay_error,
            replay_error_pose, replay_error_contact,
            np.stack([left_replay, right_replay]))


def collect(rows, base_cache, replay, xml, project):
    metadata = json.loads(str(base_cache["metadata_json"]))
    index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(metadata)}
    wanted = {(r["episode_id"], r["chunk_start"]): r for r in rows}
    episodes = sorted({r["episode_id"] for r in rows}, key=int)
    coefficients = base_cache["coefficients"]
    endpoints = base_cache["endpoints"]
    sim = DirectJengaSim(xml)
    records, raw_responses, new_coefficients, replayed_endpoints = [], [], [], []
    try:
        for number, episode_id in enumerate(episodes):
            episode = replay.episode(episode_id)
            sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                key = (episode_id, start)
                if key in wanted:
                    row = wanted[key]
                    i = index[key]
                    split = row["local_margin"]["split_index"]
                    chunk = episode.actions[start:start + HORIZON]
                    (distances, extra_coeff, extra_response, intervals, replay_error,
                     pose_error, contact_error, replayed) = refine_one(
                        sim, sim.snapshot(), chunk,
                        coefficients[split - 1], coefficients[split],
                        endpoints[i, split - 1], endpoints[i, split], project)
                    records.append({"episode_id": episode_id, "chunk_start": start,
                                    "stratum": row["stratum"],
                                    "distances_by_tail": distances.tolist(),
                                    "cached_endpoint_replay_error": replay_error,
                                    "cached_pose_replay_error": pose_error,
                                    "cached_contact_replay_error": contact_error,
                                    "refined_coefficients": extra_coeff.tolist(),
                                    "intervals": intervals})
                    raw_responses.append(extra_response)
                    new_coefficients.append(extra_coeff)
                    replayed_endpoints.append(replayed)
                sim.execute(action)
            print(f"  refine {number + 1}/{len(episodes)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return (records, np.stack(raw_responses), np.stack(new_coefficients),
            np.stack(replayed_endpoints))


def summarise(rows):
    result = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == stratum]
        result[stratum] = {"candidates": len(part),
                           "maximum_cached_endpoint_replay_error": float(max(
                               r["cached_endpoint_replay_error"] for r in part)),
                           "maximum_cached_pose_replay_error": float(max(
                               r["cached_pose_replay_error"] for r in part)),
                           "maximum_cached_contact_replay_error": float(max(
                               r["cached_contact_replay_error"] for r in part)),
                           "persistent_separation": int(sum(
                               r["scale_test"]["status"] == "persistent_separation"
                               for r in part)),
                           "oracle_bracketed": int(sum(
                               r["oracle_margin"]["status"] == "bracketed" for r in part)),
                           "persistent_and_oracle": int(sum(
                               r["scale_test"]["status"] == "persistent_separation"
                               and r["oracle_margin"]["status"] == "bracketed" for r in part)),
                           "median_retained_fraction_tail5": float(np.median([
                               r["scale_test"]["retained_fraction_by_tail"][0]
                               for r in part])),
                           "median_retained_fraction_tail10": float(np.median([
                               r["scale_test"]["retained_fraction_by_tail"][1]
                               for r in part]))}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--base-cache", default=str(ROOT / "results/jenga/persistent_branch_margin_cache.npz"))
    ap.add_argument("--local-result", default=str(ROOT / "results/jenga/local_persistent_branch.json"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/branch_scale_refine_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/branch_scale_refine.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    local = json.loads(Path(args.local_result).read_text())
    candidates = [r for r in local["rows"] if r["local_margin"]["status"] == "bracketed"]
    base_cache = np.load(args.base_cache, allow_pickle=False)
    excluded = {r["episode_id"] for r in local["rows"]}
    scale, mean, components, reference_count = fit_unlabeled_projection(
        args.original_cache, excluded)

    def project(response):
        return (np.asarray(response) / scale - mean) @ components.T

    if args.reuse_cache:
        cached = np.load(args.cache, allow_pickle=False)
        records = json.loads(str(cached["records_json"]))
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_branch_refine_") as temp:
                records, raw_responses, extra_coefficients, replayed_endpoints = collect(
                    candidates, base_cache, replay, extract_sim(args.sim_archive, temp), project)
        finally:
            replay.close()
        target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, records_json=np.asarray(json.dumps(records)),
                            extra_responses=raw_responses, extra_coefficients=extra_coefficients,
                            replayed_endpoints=replayed_endpoints)
    by_key = {(r["episode_id"], r["chunk_start"]): r for r in candidates}
    rows = []
    for record in records:
        key = (record["episode_id"], record["chunk_start"])
        source = by_key[key]
        test = compare_scales(np.asarray(record["distances_by_tail"]))
        rows.append({"episode_id": key[0], "chunk_start": key[1],
                     "stratum": record["stratum"],
                     "candidate_margin": source["local_margin"],
                     "oracle_margin": source["oracle_margin"],
                     "refined_coefficients": record["refined_coefficients"],
                     "intervals": record["intervals"],
                     "cached_endpoint_replay_error": record["cached_endpoint_replay_error"],
                     "cached_pose_replay_error": record["cached_pose_replay_error"],
                     "cached_contact_replay_error": record["cached_contact_replay_error"],
                     "scale_test": {"status": test.status,
                                    "distances_by_tail": test.distances_by_tail,
                                    "constant_rss_by_tail": test.constant_rss_by_tail,
                                    "shrinking_rss_by_tail": test.shrinking_rss_by_tail,
                                    "retained_fraction_by_tail": test.retained_fraction_by_tail}})
    result = {"protocol": {"selected_from": args.local_result,
                            "candidates": len(candidates), "refinement_levels": 2,
                            "held_tails": list(TAILS),
                            "reference_original_states": reference_count,
                            "projection": "same frozen unlabeled PCA6 as local test",
                            "decision": "constant separation model must beat width-proportional "
                                        "shrink model at both tails; equal one-parameter complexity",
                            "oracle_used_only_after_decision": True,
                            "warning": "refines only the nearest BIC candidate per state; rejection "
                                       "does not rule out a farther genuine branch"},
              "summary": summarise(rows), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
