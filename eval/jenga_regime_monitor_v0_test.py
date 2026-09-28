"""One-time verifier-gated ground-truth TEST evaluation of frozen Regime Monitor v0.

This script intentionally supports only the frozen Jenga TEST split and refuses to overwrite its
result. It verifies the v0 protocol, benchmark, simulator/cache source hashes, and exact LMDB action
chunks before evaluating. Outcome classes are joined only after every monitor decision.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, nearest_cross_branch_pairs,  # noqa: E402
                                   result_dict as branch_result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,  # noqa: E402
                                 whole_trajectory_consequence_alarm)
from jenga_action_boundary_refine import PAIRS, _refine_state  # noqa: E402
from jenga_action_branch import action_errors, pose_features, summarise  # noqa: E402
from jenga_bench import (BENCH_FILE, EVAL_CONFIG, MANIFEST_FILE as BENCH_MANIFEST,  # noqa: E402
                         SPLITS, verify as verify_benchmark)
from jenga_generic_regime_audit import reconstruct_endpoints, rollout_trace  # noqa: E402
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_regime_monitor_v0_freeze import (MANIFEST_FILE as V0_MANIFEST,  # noqa: E402
                                            sha256_file, verify as verify_v0)
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT  # noqa: E402

SPLIT = "test"
SCALE = 1.0
OUTPUT = ROOT / "results/jenga/regime_monitor_v0/test_ground_truth.json"


def verify_test_sources(sim_archive):
    """Verify every external TEST input against the original benchmark freeze."""
    manifest = verify_benchmark()
    frozen = json.loads(BENCH_MANIFEST.read_text())["sources_sha256"]
    required = (SPLITS[SPLIT]["stage0"], SPLITS[SPLIT]["cache"],
                "results/jenga/holdout3_stage2_shared.json", "vendor/panda_express_sim.tar")
    changed = []
    for name in required:
        actual = sha256_file(ROOT / name)
        if actual != frozen.get(name):
            changed.append((name, frozen.get(name), actual))
    archive = Path(sim_archive).resolve()
    if archive != (ROOT / "vendor/panda_express_sim.tar").resolve():
        raise SystemExit("v0 TEST requires the frozen simulator archive path")
    if changed:
        details = "\n".join(f"  {name}: expected {expected}, now {actual}"
                            for name, expected, actual in changed)
        raise SystemExit("frozen TEST source mismatch:\n" + details)
    return manifest


def prepare():
    """Discover candidates from TEST trajectories without using their grading classes."""
    bench = np.load(BENCH_FILE, allow_pickle=False)
    spec = SPLITS[SPLIT]
    metadata = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
    cache_index = {(row["episode_id"], row["chunk_start"]): i
                   for i, row in enumerate(metadata)}
    cache = np.load(ROOT / spec["cache"], allow_pickle=False)
    scale_index = EVAL_CONFIG["scales"].index(SCALE)
    snippets = cache["snippets"]
    if not np.array_equal(snippets, bench["snippets"]):
        raise SystemExit("TEST cache snippets differ from the frozen benchmark")
    rows, wanted = [], {}
    early_stop = RECORD_AT.index(10) + 1
    for i, episode_id in enumerate(bench["test_episode"]):
        episode_id = str(episode_id)
        chunk = int(bench["test_chunk"][i])
        cache_row = cache_index[(episode_id, chunk)]
        pose = cache["pose"][cache_row, scale_index]
        windows = bench["test_windows"][i, scale_index]
        features = pose_features(pose)
        initial = smooth_vs_branch_alarm(
            action_errors(windows), features[:, :early_stop], features)
        row = {
            "split": SPLIT, "episode_id": episode_id, "chunk_start": chunk,
            "scale": str(SCALE),
            # Grading fields are copied only after the alarm above has been computed.
            "class": str(bench["test_classes"][i, scale_index]),
            "topple_count": int(bench["test_topples"][i, scale_index]),
            "initial_alarm": bool(initial.alarm), "boundary_refined_alarm": False,
            "alarm": False, "initial_evidence": branch_result_dict(initial),
        }
        rows.append(row)
        if not initial.alarm:
            continue
        coordinates, labels = action_branch_partition(
            action_errors(windows), features[:, :early_stop], features)
        pairs = nearest_cross_branch_pairs(coordinates, labels, PAIRS)
        wanted.setdefault(episode_id, {})[chunk] = {
            "pose": pose, "labels": labels, "pairs": pairs,
            "probe_chunks": windows[:, 2:2 + HORIZON], "row": len(rows) - 1}
    return rows, wanted, snippets


def _run_episode(job):
    episode_id, targets, lmdb, xml, snippets, reset_base = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id) + int(reset_base))
        for step, action in enumerate(episode.actions):
            if step in targets:
                target = targets[step]
                chunk = np.asarray(episode.actions[step:step + HORIZON], np.float32)
                reconstructed = np.repeat(chunk[None], len(snippets), axis=0)
                reconstructed[:, :, :3] -= snippets
                if not np.array_equal(reconstructed, target["probe_chunks"]):
                    raise RuntimeError(f"LMDB actions differ from frozen TEST windows at "
                                       f"episode {episode_id}, chunk {step}")
                snapshot = sim.snapshot()
                start_pose = all_block_pose(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                boundary, pair_details = _refine_state(
                    sim, snapshot, chunk, snippets, target["pose"], target["labels"],
                    target["pairs"], start_pose, start_tilt)
                endpoints = reconstruct_endpoints(snippets, pair_details)
                traces = np.stack([[rollout_trace(sim, snapshot, chunk, noise)
                                    for noise in pair] for pair in endpoints])
                evidence = whole_trajectory_consequence_alarm(
                    traces[:, :, :, :45], endpoints[:, 0] - endpoints[:, 1],
                    boundary["early_delta_bic"], boundary["full_delta_bic"])
                out[step] = {
                    "boundary_refined_alarm": bool(boundary["alarm"]),
                    "boundary_evidence": boundary, "pair_details": pair_details,
                    **consequence_result_dict(evidence),
                }
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def evaluate(lmdb, xml, workers):
    rows, wanted, snippets = prepare()
    jobs = [(episode_id, targets, lmdb, xml, snippets,
             SPLITS[SPLIT]["reset_base"]) for episode_id, targets in wanted.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                rows[wanted[episode_id][step]["row"]].update(result)
            print(f"v0 TEST episode {number}/{len(jobs)}", flush=True)
    return rows


def candidate_summary(rows):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    return {name: {
        "states": sum(row["class"] == name for row in rows),
        "initial_candidates": sum(row["class"] == name and row["initial_alarm"] for row in rows),
        "boundary_candidates": sum(row["class"] == name and row["boundary_refined_alarm"]
                                   for row in rows),
        "v0_alarms": sum(row["class"] == name and row["alarm"] for row in rows),
    } for name in classes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() != OUTPUT.resolve():
        raise SystemExit(f"one-time v0 TEST result must be written to {OUTPUT}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite one-time v0 TEST result: {output}")
    v0 = verify_v0(V0_MANIFEST)
    benchmark = verify_test_sources(args.sim_archive)
    with tempfile.TemporaryDirectory(prefix="jenga_v0_test_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        rows = evaluate(args.lmdb, xml, args.workers)
    summary = summarise(rows, [SCALE])
    result = {
        "protocol": {
            "id": "regime-monitor-v0", "split": "TEST, one-time frozen evaluation",
            "calibration_free": True, "task_information": "classes joined for grading only",
            "overwrite_policy": "immutable; evaluator refuses overwrite",
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime_monitor_v0_sha256": v0["protocol_sha256"],
        "benchmark_sha256": benchmark["bench_sha256"],
        "test_source_sha256": {name: sha256_file(ROOT / name) for name in (
            SPLITS[SPLIT]["stage0"], SPLITS[SPLIT]["cache"],
            "results/jenga/holdout3_stage2_shared.json", "vendor/panda_express_sim.tar")},
        "summary": summary,
        "candidate_summary": candidate_summary(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"protocol_sha256": v0["protocol_sha256"],
                      "candidate_summary": result["candidate_summary"],
                      "alarms": summary["test"][str(SCALE)]}, indent=2))


if __name__ == "__main__":
    main()
