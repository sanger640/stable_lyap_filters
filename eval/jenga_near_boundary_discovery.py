"""Construct label-free near-boundary Jenga action families by broad probing.

Failure labels are used only to construct and evaluate a stress-test set.  The
detector never sees them.  A broad physical probe locates the nearest outcome
transition, then an operational eps=.10 family is recentered at that action.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import probe_scalars  # noqa: E402
from jenga_neighbor_validation import choose_pools, prior_keys  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, TOPPLE_DEG,
                                    extract_sim)  # noqa: E402

TAIL = 5


def offset_probe_chunks(chunk, scalars, eps, center_offset=0.0):
    chunk = np.asarray(chunk, np.float32); z = np.asarray(scalars, np.float32)
    direction = chunk - chunk[:1]; direction[:, 3] = 0.0
    coefficient = float(center_offset) + float(eps) * z
    probes = chunk[None] + coefficient[:, None, None] * direction[None]
    probes[:, :, 3] = chunk[None, :, 3]
    return probes


def neighbor_outcomes(sim, snapshot, probes):
    outcomes = []
    for probe in probes:
        sim.restore(snapshot)
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in probe:
            sim.execute(action, trace)
        for _ in range(TAIL):
            sim.execute(probe[-1], trace)
        outcomes.append(bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG))
    sim.restore(snapshot)
    return np.asarray(outcomes, bool)


def nearest_transition(scalars, outcomes):
    changed = np.flatnonzero(np.asarray(outcomes[1:]) != np.asarray(outcomes[:-1]))
    if not len(changed):
        return None
    midpoints = (np.asarray(scalars)[changed] + np.asarray(scalars)[changed + 1]) / 2
    return float(midpoints[np.argmin(np.abs(midpoints))])


def discover(candidates, replay, xml, broad_scalars, local_scalars, broad_eps, local_eps):
    wanted = {}
    for row in candidates:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    results = []; sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted[episode_id]:
                    snapshot = sim.snapshot(); chunk = episode.actions[start:start + HORIZON]
                    broad = neighbor_outcomes(
                        sim, snapshot, offset_probe_chunks(chunk, broad_scalars, broad_eps))
                    transition = nearest_transition(broad_scalars, broad)
                    row = dict(wanted[episode_id][start])
                    row.update({"broad_toppled": int(broad.sum()),
                                "broad_transition_scalar": transition,
                                "center_offset": None, "local_toppled": None,
                                "supported_boundary": False})
                    if transition is not None:
                        offset = float(broad_eps * transition)
                        local = neighbor_outcomes(sim, snapshot, offset_probe_chunks(
                            chunk, local_scalars, local_eps, offset))
                        toppled = int(local.sum())
                        row.update({"center_offset": offset, "local_toppled": toppled,
                                    "supported_boundary": bool(
                                        min(toppled, len(local) - toppled) >= 8)})
                    results.append(row)
                sim.execute(action)
            print(f"  discover {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--scan-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--holdout-cache", default=str(ROOT / "results/jenga/ordered_holdout_spread_cache.npz"))
    ap.add_argument("--validation-cache", default=str(ROOT / "results/jenga/neighbor_validation_cache.npz"))
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--broad-probes", type=int, default=21); ap.add_argument("--broad-eps", type=float, default=.30)
    ap.add_argument("--local-probes", type=int, default=50); ap.add_argument("--local-eps", type=float, default=.10)
    ap.add_argument("--output", default=str(ROOT / "results/jenga/near_boundary_discovery.json"))
    args = ap.parse_args()
    excluded = prior_keys(args.original_cache, args.holdout_cache, args.validation_cache)
    _, candidates = choose_pools(args.scan_cache, excluded, 1, args.candidates)
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_near_boundary_") as temp:
            rows = discover(candidates, replay, extract_sim(args.sim_archive, temp),
                            probe_scalars(args.broad_probes), probe_scalars(args.local_probes),
                            args.broad_eps, args.local_eps)
    finally:
        replay.close()
    result = {"protocol": {"candidates": len(rows), "broad_probes": args.broad_probes,
                           "broad_eps": args.broad_eps, "local_probes": args.local_probes,
                           "local_eps": args.local_eps, "minimum_each": 8},
              "broad_mixed": int(sum(row["broad_transition_scalar"] is not None for row in rows)),
              "supported_local_boundaries": int(sum(row["supported_boundary"] for row in rows)),
              "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
