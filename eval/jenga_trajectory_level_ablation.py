"""2x2 DEV oracle ablation: D2/physical boundary scales x D2/physical consequences."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import boundary_refinement_alarm, result_dict  # noqa: E402
from consequence_monitor import (result_dict as consequence_result_dict,  # noqa: E402
                                 whole_trajectory_consequence_alarm)
from jenga_action_boundary_refine import _projector  # noqa: E402
from jenga_action_branch import pose_features, sampled_predicted_pose  # noqa: E402
from jenga_bench import (BENCH_FILE, EVAL_CONFIG, sha256_file,  # noqa: E402
                         verify as verify_benchmark)
from jenga_regime_monitor_v0_freeze import verify as verify_v0  # noqa: E402
from jenga_selective_router_dev import nested_noises  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_w5_eval import load_model, predict  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402

SAMPLE_INDICES = (12, 17, 27, 36, 37)  # Hold 5, 10, 20, 29, 30 after H=8.


def simulate_endpoint(sim, snapshot, chunk, noise):
    sim.restore(snapshot)
    perturbed = np.asarray(chunk, np.float32).copy()
    perturbed[:, :3] -= noise
    states = []
    for action in perturbed:
        sim.execute(action)
        states.append(step_state(sim))
    for _ in range(30):
        sim.execute(chunk[-1])
        states.append(step_state(sim))
    return np.stack(states)


def _simulate_episode(job):
    episode_id, targets, lmdb, xml = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    output = {}
    try:
        sim.reset(int(episode_id))
        for step, action in enumerate(episode.actions):
            if step in targets:
                snapshot = sim.snapshot()
                chunk = episode.actions[step:step + 8]
                pair_traces = []
                for pair_levels in targets[step]:
                    levels = [[simulate_endpoint(sim, snapshot, chunk, noise)
                               for noise in endpoint] for endpoint in pair_levels]
                    pair_traces.append(levels)
                output[step] = np.asarray(pair_traces, np.float32)
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, output


def collect_dense(rows, snippets, lmdb, sim_archive, workers):
    targets = {}
    for row in rows:
        if row["initial_alarm"]:
            targets.setdefault(row["episode_id"], {})[row["chunk_start"]] = [
                nested_noises(snippets, pair) for pair in row["pair_details"]]
    with tempfile.TemporaryDirectory(prefix="jenga_trajectory_ablation_") as temp:
        xml = str(extract_sim(sim_archive, temp))
        jobs = [(episode, values, lmdb, xml) for episode, values in targets.items()]
        output = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for number, (episode, values) in enumerate(pool.map(_simulate_episode, jobs), 1):
                output.update({(episode, int(step)): traces for step, traces in values.items()})
                print(f"dense physical episode {number}/{len(jobs)}", flush=True)
    return output


def save_cache(path, traces, metadata):
    keys = sorted(traces, key=lambda key: (int(key[0]), key[1]))
    arrays = np.stack([traces[key] for key in keys])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=np.asarray([f"{ep}:{step}" for ep, step in keys]),
                        traces=arrays, metadata_json=np.asarray(json.dumps(metadata)))


def load_cache(path, expected):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    if metadata != expected:
        raise RuntimeError("refusing trajectory ablation cache with different provenance")
    keys = [tuple(value.split(":")) for value in data["keys"].tolist()]
    return {(episode, int(step)): trace for (episode, step), trace in zip(keys, data["traces"])}


def physical_boundary(traces, predicted_features):
    early_project = _projector(predicted_features[:, :2])
    full_project = _projector(predicted_features)
    early, full = [], []
    for pair in traces:
        pair_early, pair_full = [], []
        for level in pair:
            sampled = pose_features(level[:, SAMPLE_INDICES, :27])
            early_coordinates = early_project(sampled[:, :2])
            full_coordinates = full_project(sampled)
            pair_early.append(float(np.linalg.norm(
                early_coordinates[0] - early_coordinates[1])))
            pair_full.append(float(np.linalg.norm(
                full_coordinates[0] - full_coordinates[1])))
        early.append(pair_early); full.append(pair_full)
    return boundary_refinement_alarm(early, full), early, full


def d2_consequence_with_boundary(row, boundary):
    pair_boundary = ((np.asarray(boundary.early_delta_bic) > 0) &
                     (np.asarray(boundary.full_delta_bic) > 0))
    commitment = np.asarray(row["commitment_delta_bic"]) > 0
    persistence = np.asarray(row["persistence_delta_bic"]) > 0
    consequential = pair_boundary & commitment & persistence
    return {
        "alarm": bool(consequential.sum() >= 2),
        "consequential_pairs": int(consequential.sum()),
        "boundary_pairs": int(pair_boundary.sum()),
        "commitment_pairs": int(commitment.sum()),
        "persistence_pairs": int(persistence.sum()),
    }


def evaluate_row(row, traces, predicted_features, snippets):
    if not row["initial_alarm"]:
        empty = {"alarm": False, "boundary_refined_alarm": False,
                 "boundary_pairs": 0, "commitment_pairs": 0, "persistence_pairs": 0}
        return {name: empty.copy() for name in (
            "d2_boundary_d2_consequence", "physical_boundary_d2_consequence",
            "d2_boundary_physical_consequence", "physical_boundary_physical_consequence")}

    d2_early = [pair["early_gaps"] for pair in row["pair_details"]]
    d2_full = [pair["full_gaps"] for pair in row["pair_details"]]
    d2_boundary = boundary_refinement_alarm(d2_early, d2_full)
    physical, physical_early, physical_full = physical_boundary(traces, predicted_features)
    final_traces = traces[:, -1]
    action_differences = np.stack([
        nested_noises(snippets, pair)[-1, 0] - nested_noises(snippets, pair)[-1, 1]
        for pair in row["pair_details"]])
    d2_physical = whole_trajectory_consequence_alarm(
        final_traces[:, :, :, :45], action_differences,
        d2_boundary.early_delta_bic, d2_boundary.full_delta_bic)
    physical_physical = whole_trajectory_consequence_alarm(
        final_traces[:, :, :, :45], action_differences,
        physical.early_delta_bic, physical.full_delta_bic)

    def packed(boundary, consequence, source):
        value = consequence_result_dict(consequence) if source == "physical" else consequence
        return {**value, "boundary_refined_alarm": bool(boundary.alarm),
                "boundary_evidence": result_dict(boundary)}

    return {
        "d2_boundary_d2_consequence": packed(
            d2_boundary, d2_consequence_with_boundary(row, d2_boundary), "cached"),
        "physical_boundary_d2_consequence": packed(
            physical, d2_consequence_with_boundary(row, physical), "cached"),
        "d2_boundary_physical_consequence": packed(d2_boundary, d2_physical, "physical"),
        "physical_boundary_physical_consequence": packed(
            physical, physical_physical, "physical"),
        "physical_early_gaps": np.asarray(physical_early).tolist(),
        "physical_full_gaps": np.asarray(physical_full).tolist(),
    }


def summarize(rows, decisions, arm):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    output = {}
    for class_name in classes:
        selected = [(row, value[arm]) for row, value in zip(rows, decisions)
                    if row["class"] == class_name]
        output[class_name] = {
            "states": len(selected),
            "initial_candidates": sum(row["initial_alarm"] for row, _ in selected),
            "boundary_candidates": sum(value["boundary_refined_alarm"] for _, value in selected),
            "commitment_majority": sum(value.get("commitment_pairs", 0) >= 2
                                       for _, value in selected),
            "persistence_majority": sum(value.get("persistence_pairs", 0) >= 2
                                        for _, value in selected),
            "alarms": sum(value["alarm"] for _, value in selected),
        }
    return output


def pair_overlap(rows, decisions):
    """Compare D2 and physical signs for the exact same selected pairs."""
    classes = ("all", "topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    output = {}
    for class_name in classes:
        selected = [(row, value) for row, value in zip(rows, decisions)
                    if row["initial_alarm"] and
                    (class_name == "all" or row["class"] == class_name)]
        channels = {}
        for channel in ("boundary", "commitment", "persistence"):
            left, right = [], []
            for row, value in selected:
                if channel == "boundary":
                    d2 = value["d2_boundary_d2_consequence"]["boundary_evidence"]
                    physical = value["physical_boundary_physical_consequence"][
                        "boundary_evidence"]
                    left.extend((np.asarray(d2["early_delta_bic"]) > 0) &
                                (np.asarray(d2["full_delta_bic"]) > 0))
                    right.extend((np.asarray(physical["early_delta_bic"]) > 0) &
                                 (np.asarray(physical["full_delta_bic"]) > 0))
                elif channel == "commitment":
                    left.extend(np.asarray(row["commitment_delta_bic"]) > 0)
                    right.extend(np.asarray(value["d2_boundary_physical_consequence"][
                        "commitment_delta_bic"]) > 0)
                else:
                    left.extend(np.asarray(row["persistence_delta_bic"]) > 0)
                    right.extend(np.asarray(value["d2_boundary_physical_consequence"][
                        "persistence_delta_bic"]) > 0)
            left, right = np.asarray(left, bool), np.asarray(right, bool)
            channels[channel] = {
                "pairs": int(len(left)), "d2_positive": int(left.sum()),
                "physical_positive": int(right.sum()),
                "both_positive": int((left & right).sum()),
                "d2_only": int((left & ~right).sum()),
                "physical_only": int((~left & right).sum()),
                "both_negative": int((~left & ~right).sum()),
                "agreement": float(np.mean(left == right)) if len(left) else None,
            }
        output[class_name] = channels
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/trajectory_level_ablation_protocol.json"))
    parser.add_argument("--base", default=str(
        ROOT / "results/jenga/regime_monitor_v0/selective_router_dev_base/w6_cw_d2_s1.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--cache", default=str(
        ROOT / "results/jenga/trajectory_level_ablation_cache.npz"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/trajectory_level_ablation_summary.json"))
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    verify_benchmark(); verify_v0()
    base = json.loads(Path(args.base).read_text())
    rows = base["rows"]
    bench = np.load(BENCH_FILE, allow_pickle=False)
    snippets = bench["snippets"]
    metadata = {
        "protocol_sha256": sha256_file(args.protocol),
        "base_sha256": sha256_file(args.base),
        "benchmark_sha256": sha256_file(BENCH_FILE),
    }
    if args.reuse_cache:
        traces = load_cache(args.cache, metadata)
    else:
        traces = collect_dense(rows, snippets, args.lmdb, args.sim_archive, args.workers)
        save_cache(args.cache, traces, metadata)

    model, model_scale = load_model(args.checkpoint, args.device)
    index = {(str(ep), int(chunk)): i for i, (ep, chunk) in enumerate(
        zip(bench["dev_episode"], bench["dev_chunk"]))}
    decisions = []
    for number, row in enumerate(rows, 1):
        if not row["initial_alarm"]:
            decisions.append(evaluate_row(row, None, None, snippets))
            continue
        key = (row["episode_id"], row["chunk_start"])
        i = index[key]
        windows = bench["dev_windows"][i, EVAL_CONFIG["scales"].index(1.0)]
        predicted = predict(model, model_scale, bench["dev_start"][i], windows, args.device)
        features = sampled_predicted_pose(predicted, model)
        decisions.append(evaluate_row(row, traces[key], features, snippets))
        if number % 20 == 0:
            print(f"evaluated {number}/{len(rows)}", flush=True)

    arms = (
        "d2_boundary_d2_consequence", "physical_boundary_d2_consequence",
        "d2_boundary_physical_consequence", "physical_boundary_physical_consequence")
    summaries = {arm: summarize(rows, decisions, arm) for arm in arms}
    reference = protocol["reference_gate"]
    reference_pass = {arm: (
        value["topple_fork"]["alarms"] >= reference["topple_alarms_min"] and
        value["quiet"]["alarms"] <= reference["quiet_alarms_max"])
        for arm, value in summaries.items()}
    identity_exact = all(
        bool(row["alarm"]) == bool(value["d2_boundary_d2_consequence"]["alarm"])
        for row, value in zip(rows, decisions))
    if not identity_exact:
        raise RuntimeError("D2/D2 ablation arm does not reproduce frozen baseline decisions")
    output = {
        "protocol": protocol, "protocol_sha256": sha256_file(args.protocol),
        "base_sha256": sha256_file(args.base), "cache_metadata": metadata,
        "identity_exact": identity_exact, "summaries": summaries,
        "reference_gate_pass": reference_pass,
        "pair_overlap": pair_overlap(rows, decisions), "decisions": decisions,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "reference_gate_pass": reference_pass,
                      "identity_exact": identity_exact}, indent=2))


if __name__ == "__main__":
    main()
