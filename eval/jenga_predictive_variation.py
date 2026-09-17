"""Compare detected future splits with controller-error-induced variation.

This is an effect-size diagnostic, not a threshold fitted to failures. Residual
tracking-error snippets are injected into action targets as a proxy for execution
variation; that proxy is not a validated stochastic controller model.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from action_uncertainty import tracking_arrays  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_predictive_regime_probe import (HORIZON, all_block_pose,
                                           simulate_probe)  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402


def residual_snippets(replay, excluded, coefficients, count=8, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for ep in replay.episode_ids:
        if ep in excluded:
            continue
        design, errors = tracking_arrays([replay.episode(ep)])
        residual = errors - design @ coefficients
        if len(residual) >= HORIZON:
            pool.append(residual)
    selected = []
    for _ in range(count):
        residual = pool[int(rng.integers(len(pool)))]
        start = int(rng.integers(len(residual) - HORIZON + 1))
        selected.append(residual[start:start + HORIZON])
    return np.asarray(selected, np.float32)


def simulate_jitter(sim, snapshot, chunk, jitter, start):
    sim.restore(snapshot)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= jitter
    for action in actions:
        sim.execute(action)
    frames = [all_block_pose(sim) - start]
    for held in range(1, 11):
        sim.execute(chunk[-1])
        if held in (5, 10):
            frames.append(all_block_pose(sim) - start)
    sim.restore(snapshot)
    return np.stack(frames)


def collect(rows, replay, xml, snippets):
    wanted = {(r["episode_id"], r["chunk_start"]): r for r in rows}
    episodes = sorted({r["episode_id"] for r in rows}, key=int)
    sim = DirectJengaSim(xml)
    metadata, outcomes = [], []
    try:
        for number, ep in enumerate(episodes):
            episode = replay.episode(ep)
            sim.reset(int(ep))
            for index, action in enumerate(episode.actions):
                key = (ep, index)
                if key in wanted:
                    chunk = episode.actions[index:index + HORIZON]
                    snapshot = sim.snapshot()
                    start = all_block_pose(sim)
                    outcomes.append([simulate_jitter(sim, snapshot, chunk, noise, start)
                                     for noise in snippets])
                    metadata.append(wanted[key])
                sim.execute(action)
            print(f"  jitter replay {number + 1}/{len(episodes)} ep{ep}", flush=True)
    finally:
        sim.close()
    return metadata, np.asarray(outcomes, np.float32)


def effect_sizes(row, original, jitter):
    scale = robust_component_scale(original)
    flat = (original / scale).reshape(-1, original.shape[-1])
    mean = flat.mean(0)
    components = np.linalg.svd(flat - mean, full_matrices=False)[2][:6]
    projected_original = (original / scale - mean) @ components.T
    projected_jitter = (jitter / scale - mean) @ components.T
    pair_i, pair_j = np.triu_indices(len(jitter), k=1)
    outcome = {}
    for time, name in ((1, "tail5"), (2, "tail10")):
        spread = np.linalg.norm(projected_jitter[pair_i, time]
                                - projected_jitter[pair_j, time], axis=1)
        baseline = float(np.median(spread))
        outcome[name] = {"jitter_pairwise_median": baseline,
                         "jitter_pairwise_p90": float(np.quantile(spread, .90))}
        if row["detector"]["alarm"]:
            axis = row["detector"]["margin"]["axis"]
            split = row["detector"]["directions"][axis][name]["split_index"]
            gap = float(np.linalg.norm(projected_original[axis, split, time]
                                       - projected_original[axis, split - 1, time]))
            outcome[name]["detected_gap_distance"] = gap
            outcome[name]["gap_to_jitter_median_ratio"] = (
                gap / baseline if baseline > 1e-12 else None)
    return outcome


def summarise(rows):
    result = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == stratum]
        alarmed = [r for r in part if r["detector_alarm"]]
        ratios = [r["effect_size"]["tail10"]["gap_to_jitter_median_ratio"]
                  for r in alarmed if r["effect_size"]["tail10"]["gap_to_jitter_median_ratio"]
                  is not None]
        result[stratum] = {"states": len(part), "alarms": len(alarmed),
                           "tail10_median_gap_to_jitter_ratio_on_alarms": (
                               float(np.median(ratios)) if ratios else None),
                           "alarms_with_zero_jitter_spread": len(alarmed) - len(ratios)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--probe-result", default=str(ROOT / "results/jenga/predictive_regime_probe.json"))
    ap.add_argument("--probe-cache", default=str(ROOT / "results/jenga/predictive_regime_probe_cache.npz"))
    ap.add_argument("--uncertainty", default=str(ROOT / "results/jenga/tracking_uncertainty.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/predictive_variation_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/predictive_variation.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    probe_result = json.loads(Path(args.probe_result).read_text())
    probe_cache = np.load(args.probe_cache, allow_pickle=False)
    uncertainty = json.loads(Path(args.uncertainty).read_text())
    excluded = {r["episode_id"] for r in probe_result["rows"]}
    replay = JengaReplay(args.lmdb)
    try:
        snippets = residual_snippets(replay, excluded,
                                     np.asarray(uncertainty["lag_coefficients"]))
        if args.reuse_cache:
            cache = np.load(args.cache, allow_pickle=False)
            metadata = json.loads(str(cache["metadata_json"]))
            outcomes = cache["responses"]
        else:
            with tempfile.TemporaryDirectory(prefix="jenga_predictive_variation_") as temp:
                metadata, outcomes = collect(probe_result["rows"], replay,
                                             extract_sim(args.sim_archive, temp), snippets)
            target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, metadata_json=np.asarray(json.dumps(metadata)),
                                responses=outcomes, injected_snippets=snippets)
    finally:
        replay.close()
    original_meta = json.loads(str(probe_cache["metadata_json"]))
    original_by_key = {(r["episode_id"], r["chunk_start"]): i
                       for i, r in enumerate(original_meta)}
    rows = []
    for meta, jitter in zip(metadata, outcomes):
        key = (meta["episode_id"], meta["chunk_start"])
        i = original_by_key[key]
        original = probe_cache["responses"][i]
        rows.append({"episode_id": key[0], "chunk_start": key[1],
                     "stratum": meta["stratum"],
                     "detector_alarm": meta["detector"]["alarm"],
                     "oracle_mixed": meta["oracle"]["mixed_directions"] > 0,
                     "effect_size": effect_sizes(meta, original, jitter)})
    result = {"protocol": {"injected_repeats": len(snippets),
                            "tracking_snippets_from_test_excluded_episodes": True,
                            "injection": "subtract centered tracking residual from each "
                                         "commanded target during H=8; hold nominal target afterward",
                            "effect_size": "detected adjacent future-pose difference divided by "
                                           "median pairwise jitter-rollout difference in same PCA6",
                            "no_threshold_or_alarm_change": True,
                            "warning": "tracking-error injection is an unvalidated execution "
                                       "proxy, not real repeated hardware trials"},
              "summary": summarise(rows), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
