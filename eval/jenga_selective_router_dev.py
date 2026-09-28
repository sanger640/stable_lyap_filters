"""Prospective DEV evaluation of TRAIN-fitted forced and selective scale-curve routers."""
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
from jenga_action_boundary_refine import _projector  # noqa: E402
from jenga_action_branch import pose_features, sampled_predicted_pose  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, verify as verify_benchmark  # noqa: E402
from jenga_causal_router import (causal_features, load_alphas,  # noqa: E402
                                 standardize_fit, train_router as train_forced)
from jenga_oracle_routing_upper_bound import (balanced_kmeans, load_groups, log_gap_curve,  # noqa: E402
                                              predict_groups, sha256, validation_ids)
from jenga_regime_monitor_v0_freeze import verify as verify_v0  # noqa: E402
from jenga_selective_cost_router import (option_cost_components, normalized_option_cost,  # noqa: E402
                                          train_router as train_selective)
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import rollout as physical_rollout  # noqa: E402
from jenga_w5_eval import load_model, predict  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402


def pair_feature(start, windows, snippets, pair):
    """Causal feature matching D3's nominal/residual convention for one DEV pair."""
    left, right = pair
    nominal = np.asarray(windows[0, 2:10], np.float32).copy()
    nominal[:, :3] += snippets[0]
    residual = 2.0 * (windows[right, 2:10, :3] - windows[left, 2:10, :3])
    return np.concatenate([np.asarray(start, np.float32), nominal.reshape(-1), residual.reshape(-1)])


def corrected_gaps(gaps, correction, phase):
    values = np.asarray(gaps, float).copy()
    if values.shape != (6,) or np.asarray(correction).shape != (5, 3):
        raise ValueError("expected six gaps and a five-level by three-phase correction")
    values[1:] *= np.exp(np.asarray(correction)[:, phase])
    return values


def route_row(row, assignments, corrections):
    """Recompute boundary and final conjunction while preserving consequence evidence."""
    if not row["initial_alarm"]:
        return {"alarm": False, "boundary_refined_alarm": False, "assignments": []}
    early, full = [], []
    for pair, option in zip(row["pair_details"], assignments):
        early.append(corrected_gaps(pair["early_gaps"], corrections[option], 1))
        full.append(corrected_gaps(pair["full_gaps"], corrections[option], 2))
    boundary = boundary_refinement_alarm(early, full)
    pair_boundary = ((np.asarray(boundary.early_delta_bic) > 0) &
                     (np.asarray(boundary.full_delta_bic) > 0))
    commitment = np.asarray(row["commitment_delta_bic"]) > 0
    persistence = np.asarray(row["persistence_delta_bic"]) > 0
    consequential = pair_boundary & commitment & persistence
    return {
        "alarm": bool(consequential.sum() >= 2),
        "boundary_refined_alarm": bool(boundary.alarm),
        "assignments": list(map(int, assignments)),
        "boundary_evidence": result_dict(boundary),
        "consequential_pairs": int(consequential.sum()),
    }


def summarize(rows, decisions):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    output = {}
    for name in classes:
        selected = [(row, decision) for row, decision in zip(rows, decisions)
                    if row["class"] == name]
        output[name] = {
            "states": len(selected),
            "initial_candidates": sum(row["initial_alarm"] for row, _ in selected),
            "boundary_candidates": sum(value["boundary_refined_alarm"] for _, value in selected),
            "commitment_majority": sum(row.get("commitment_pairs", 0) >= 2
                                       for row, _ in selected),
            "persistence_majority": sum(row.get("persistence_pairs", 0) >= 2
                                        for row, _ in selected),
            "alarms": sum(value["alarm"] for _, value in selected),
        }
    return output


def nested_noises(snippets, pair):
    endpoints = [snippets[pair["probe_indices"][0]].copy(),
                 snippets[pair["probe_indices"][1]].copy()]
    levels = [[endpoints[0].copy(), endpoints[1].copy()]]
    for side in pair["midpoint_sides"]:
        midpoint = .5 * (endpoints[0] + endpoints[1])
        endpoints[int(side)] = midpoint
        levels.append([endpoints[0].copy(), endpoints[1].copy()])
    return np.asarray(levels, np.float32)


def _physical_episode(job):
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
                from jenga_predictive_regime_probe import all_block_pose
                start_pose = all_block_pose(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                chunk = episode.actions[step:step + 8]
                pair_poses = []
                for noises in targets[step]:
                    level_poses = []
                    for level in noises:
                        endpoint_poses = [physical_rollout(
                            sim, snapshot, chunk, noise, start_pose, start_tilt)[0]
                                          for noise in level]
                        level_poses.append(endpoint_poses)
                    pair_poses.append(level_poses)
                output[step] = np.asarray(pair_poses, np.float32)
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, output


def collect_physical(rows, snippets, lmdb, sim_archive, workers):
    wanted = {}
    for row in rows:
        if row["initial_alarm"]:
            wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = [
                nested_noises(snippets, pair) for pair in row["pair_details"]]
    with tempfile.TemporaryDirectory(prefix="jenga_selective_dev_") as temp:
        xml = str(extract_sim(sim_archive, temp))
        jobs = [(episode, targets, lmdb, xml) for episode, targets in wanted.items()]
        output = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for number, (episode, values) in enumerate(pool.map(_physical_episode, jobs), 1):
                output.update({(episode, int(step)): pose for step, pose in values.items()})
                print(f"physical oracle episode {number}/{len(jobs)}", flush=True)
    return output


def physical_log_curves(rows, bench, snippets, model, model_scale, device, physical):
    """Physical early/full log-gap curves in each D2 candidate's frozen projector."""
    index = {(str(ep), int(chunk)): i for i, (ep, chunk) in enumerate(
        zip(bench["dev_episode"], bench["dev_chunk"]))}
    curves = {}
    for row in rows:
        if not row["initial_alarm"]:
            continue
        key = (row["episode_id"], row["chunk_start"])
        i = index[key]
        windows = bench["dev_windows"][i, EVAL_CONFIG["scales"].index(1.0)]
        predicted = predict(model, model_scale, bench["dev_start"][i], windows, device)
        features = sampled_predicted_pose(predicted, model)
        early_project = _projector(features[:, :2])
        full_project = _projector(features)
        pair_curves = []
        for pair_pose in physical[key]:
            early_gaps, full_gaps = [], []
            for endpoint_pose in pair_pose:
                endpoint = pose_features(endpoint_pose)
                early = early_project(endpoint[:, :2])
                full = full_project(endpoint)
                early_gaps.append(float(np.linalg.norm(early[0] - early[1])))
                full_gaps.append(float(np.linalg.norm(full[0] - full[1])))
            gaps = np.stack([early_gaps, full_gaps], axis=1)
            pair_curves.append(np.log((gaps[1:] + 1e-4) / (gaps[:1] + 1e-4)))
        curves[key] = np.asarray(pair_curves)
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/selective_router_dev_protocol.json"))
    parser.add_argument("--base", default=str(
        ROOT / "results/jenga/regime_monitor_v0/selective_router_dev_base/w6_cw_d2_s1.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/selective_router_dev_summary.json"))
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-physical", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    verify_benchmark(); verify_v0()
    base = json.loads(Path(args.base).read_text())
    rows = base["rows"]
    bench = np.load(BENCH_FILE, allow_pickle=False)
    snippets = bench["snippets"]

    starts, actions, truth_trajectories, sources = load_groups(args.data)
    alphas = load_alphas(args.data)
    heldout_ids = validation_ids(args.trace)
    fit = ~np.asarray([source in heldout_ids for source in sources])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predicted_trajectories = predict_groups(
        args.checkpoint, starts, actions, args.device, 8)
    predicted_curve = log_gap_curve(predicted_trajectories, checkpoint["state_scale"].numpy())
    truth_curve = log_gap_curve(truth_trajectories, checkpoint["state_scale"].numpy())
    residual = (truth_curve - predicted_curve).reshape(len(starts), -1)
    centroids, fit_assignment, _ = balanced_kmeans(residual[fit], modes=3)
    corrections = np.concatenate([np.zeros((1, 15)), centroids], axis=0).reshape(4, 5, 3)

    train_features = causal_features(starts, actions, alphas)
    fit_x, _, feature_mean, feature_scale = standardize_fit(
        train_features[fit], train_features[:1])
    residual_mean = residual[fit].mean(0)
    residual_scale = residual[fit].std(0); residual_scale[residual_scale < 1e-6] = 1.0
    fit_residual = (residual[fit] - residual_mean) / residual_scale
    global_cost, local_cost = option_cost_components(predicted_curve, truth_curve, corrections)
    normalizers = {"global": float(global_cost[fit, 0].mean()),
                   "local": float(local_cost[fit, 0].mean())}
    option_cost = normalized_option_cost(global_cost, local_cost, normalizers)
    log_cost = np.log(np.maximum(option_cost, 1e-6))
    constant_option = int(option_cost[fit].mean(0).argmin())

    forced_models = [train_forced(fit_x, fit_assignment, fit_residual, seed, 400)
                     for seed in protocol["router_seeds"]]
    selective_models = [train_selective(fit_x, log_cost[fit], seed, 400)
                        for seed in protocol["router_seeds"]]

    state_index = {(str(ep), int(chunk)): i for i, (ep, chunk) in enumerate(
        zip(bench["dev_episode"], bench["dev_chunk"]))}
    pair_features = {}
    for row in rows:
        if not row["initial_alarm"]:
            continue
        i = state_index[(row["episode_id"], row["chunk_start"])]
        windows = bench["dev_windows"][i, EVAL_CONFIG["scales"].index(1.0)]
        pair_features[(row["episode_id"], row["chunk_start"])] = np.stack([
            pair_feature(bench["dev_start"][i], windows, snippets, pair["probe_indices"])
            for pair in row["pair_details"]])

    def assignments_for(model, forced):
        output = {}
        model.eval()
        with torch.no_grad():
            for key, values in pair_features.items():
                x = (values - feature_mean) / feature_scale
                prediction = model(torch.as_tensor(x, dtype=torch.float32))
                if forced:
                    prediction = prediction[0]
                    output[key] = 1 + prediction.argmax(1).numpy()
                else:
                    output[key] = prediction.argmin(1).numpy()
        return output

    def evaluate(assignments):
        decisions = []
        for row in rows:
            key = (row["episode_id"], row["chunk_start"])
            choices = assignments.get(key, np.zeros(0, int))
            decisions.append(route_row(row, choices, corrections))
        return {"summary": summarize(rows, decisions), "decisions": decisions}

    identity = evaluate({key: np.zeros(3, int) for key in pair_features})
    constant = evaluate({key: np.full(3, constant_option, int) for key in pair_features})
    forced = [evaluate(assignments_for(model, True)) for model in forced_models]
    selective = [evaluate(assignments_for(model, False)) for model in selective_models]

    oracle = None
    if not args.skip_physical:
        physical = collect_physical(rows, snippets, args.lmdb, args.sim_archive, args.workers)
        model, model_scale = load_model(args.checkpoint, args.device)
        physical_curves = physical_log_curves(
            rows, bench, snippets, model, model_scale, args.device, physical)
        oracle_assignments = {}
        for row in rows:
            if not row["initial_alarm"]:
                continue
            key = (row["episode_id"], row["chunk_start"])
            d2_curve = np.stack([
                np.stack([
                    np.log((np.asarray(pair["early_gaps"])[1:] + 1e-4) /
                           (pair["early_gaps"][0] + 1e-4)),
                    np.log((np.asarray(pair["full_gaps"])[1:] + 1e-4) /
                           (pair["full_gaps"][0] + 1e-4)),
                ], axis=1) for pair in row["pair_details"]])
            truth = physical_curves[key]
            candidate = d2_curve[:, None] + corrections[None, :, :, 1:]
            error = candidate - truth[:, None]
            global_pair = np.mean(np.where(np.abs(error) <= 1, .5 * error ** 2,
                                           np.abs(error) - .5), axis=(2, 3))
            candidate_full = np.concatenate([np.zeros_like(candidate[:, :, :1]), candidate], 2)
            truth_full = np.concatenate([np.zeros_like(truth[:, :1]), truth], 1)
            local_error = np.diff(candidate_full, axis=2) - np.diff(truth_full[:, None], axis=2)
            local_pair = np.mean(np.where(np.abs(local_error) <= 1, .5 * local_error ** 2,
                                          np.abs(local_error) - .5), axis=(2, 3))
            # TRAIN identity normalizers are common positive constants; averaging the two phases
            # preserves the preregistered equal global/local weighting.
            oracle_assignments[key] = np.argmin(
                global_pair / normalizers["global"] + local_pair / normalizers["local"], axis=1)
        oracle = evaluate(oracle_assignments)

    def seed_summary(results):
        fields = {}
        for class_name in ("topple_fork", "quiet"):
            values = np.asarray([result["summary"][class_name]["alarms"] for result in results])
            fields[class_name] = {"median": float(np.median(values)),
                                  "minimum": int(values.min()), "maximum": int(values.max()),
                                  "values": values.tolist()}
        return fields

    forced_summary = seed_summary(forced)
    selective_summary = seed_summary(selective)
    gate = protocol["feasibility_gate"]
    gate_result = {
        "selective_topple_alarms": selective_summary["topple_fork"]["median"] >=
        gate["selective_topple_alarms_min"],
        "selective_quiet_alarms": selective_summary["quiet"]["median"] <=
        gate["selective_quiet_alarms_max"],
        "selective_topple_not_below_forced": selective_summary["topple_fork"]["median"] >=
        forced_summary["topple_fork"]["median"],
        "selective_quiet_not_above_forced": selective_summary["quiet"]["median"] <=
        forced_summary["quiet"]["median"],
    }
    output = {
        "protocol": protocol, "protocol_sha256": sha256(args.protocol),
        "base_result": args.base, "base_result_sha256": sha256(args.base),
        "corrections": corrections.tolist(), "constant_option": constant_option,
        "identity": identity, "constant": constant,
        "forced_router_seeds": forced, "forced_router_summary": forced_summary,
        "selective_router_seeds": selective, "selective_router_summary": selective_summary,
        "future_informed_physical_cost_oracle": oracle,
        "feasibility_gate": gate_result, "passes": bool(all(gate_result.values())),
        "decision": ("selective router passes prospective DEV gate"
                     if all(gate_result.values()) else
                     "selective router fails prospective DEV gate; do not run TEST"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"identity": identity["summary"], "constant": constant["summary"],
                      "forced": forced_summary, "selective": selective_summary,
                      "oracle": None if oracle is None else oracle["summary"],
                      "gate": gate_result, "passes": output["passes"]}, indent=2))


if __name__ == "__main__":
    main()
