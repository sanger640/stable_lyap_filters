"""Evaluate multi-peak local expansion on physical, encoded, and predicted state."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,
                           load_world_model, normalise_proprio,
                           preprocess_frames)  # noqa: E402
from local_expansion import detect_local_expansion  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import (model_probe_deltas, probe_scalars,
                                      read_probe_cache)  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_ordered_change_points import confusion, physical_boundary  # noqa: E402
from jenga_ordered_holdout import binomial_interval  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, TOPPLE_DEG,
                                    extract_sim, projection_basis)  # noqa: E402

TAIL = 5
NEIGHBOR_CONTACT_INDICES = np.asarray([0, 1, 2, 4, 5, 7, 8, 10, 11])


def quaternion_rotation6(quaternion):
    q = np.asarray(quaternion, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                     2 * (x * z - y * w), 2 * (y * z + x * w)], axis=-1)


def physical_state(sim):
    blocks = sim.block_diagnostics()
    position = blocks["position"][1:].reshape(-1)
    rotation = sim.data.xmat[sim.tracked_block_ids[1:]].reshape(2, 3, 3)
    rotation6 = rotation[:, :, :2].reshape(-1)
    contacts = sim.contact_signature()[NEIGHBOR_CONTACT_INDICES].astype(float)
    return np.concatenate([position, rotation6, contacts])


def original_physical_responses(path):
    data = dict(np.load(path, allow_pickle=False))
    position = (data["physical_block_end_position"][:, :, 1:]
                - data["physical_start_block_position"][:, None, 1:]).reshape(100, 50, -1)
    end_rotation = quaternion_rotation6(data["physical_block_end_quaternion"][:, :, 1:])
    start_rotation = quaternion_rotation6(
        data["physical_start_block_quaternion"][:, None, 1:])
    rotation = (end_rotation - start_rotation).reshape(100, 50, -1)
    contact = (data["physical_contact_end"][:, :, NEIGHBOR_CONTACT_INDICES].astype(float)
               - data["physical_start_contact"][:, None, NEIGHBOR_CONTACT_INDICES].astype(float))
    return np.concatenate([position, rotation, contact], axis=-1)


def choose_controls(validation_cache, count):
    data = dict(np.load(validation_cache, allow_pickle=False))
    metadata = json.loads(str(data["metadata_json"]))
    candidates = [(i, row) for i, row in enumerate(metadata)
                  if row["stratum"] == "quiet_control"]
    first, rest, seen = [], [], set()
    for item in candidates:
        episode = item[1]["episode_id"]
        (first if episode not in seen else rest).append(item); seen.add(episode)
    selected = (first + rest)[:int(count)]
    if len(selected) < int(count):
        raise ValueError("not enough cached quiet controls")
    rows = []
    for index, row in selected:
        item = dict(row); item.update({"stratum": "quiet_control", "center_offset": 0.0,
                                      "validation_cache_index": index})
        rows.append(item)
    return rows, data


def choose_boundaries(discovery_json, count):
    data = json.loads(Path(discovery_json).read_text())
    supported = [row for row in data["rows"] if row["supported_boundary"]]
    first, rest, seen = [], [], set()
    for row in supported:
        (first if row["episode_id"] not in seen else rest).append(row)
        seen.add(row["episode_id"])
    selected = (first + sorted(rest, key=lambda row: abs(row["local_toppled"] - 25)))[:int(count)]
    if len(selected) < int(count):
        raise ValueError(f"only {len(selected)} supported discovered boundaries")
    return [dict(row, stratum="recentered_neighbor_boundary") for row in selected]


def simulate_responses(sim, snapshot, probes, render_endpoints):
    start_state = physical_state(sim); response = []; outcomes = []
    frames, props = [], []
    for probe in probes:
        sim.restore(snapshot)
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in probe:
            sim.execute(action, trace)
        for _ in range(TAIL):
            sim.execute(probe[-1], trace)
        response.append(physical_state(sim) - start_state)
        outcomes.append(bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG))
        if render_endpoints:
            frames.append(sim.render()); props.append(sim.proprio())
    sim.restore(snapshot)
    return (np.asarray(response, np.float32), np.asarray(outcomes, bool),
            np.asarray(frames), np.asarray(props))


def encode_real(model, device, start_frame, start_prop, frames, props, projection, batch_size):
    mean, components = projection
    mean_t = torch.as_tensor(mean, device=device); components_t = torch.as_tensor(components, device=device)
    with torch.inference_mode():
        start = model.encode_obs({"visual": preprocess_frames(start_frame[None, None], device),
                                  "proprio": normalise_proprio(start_prop[None, None], device)})["visual"][:, 0]
    start = ((start.float().flatten(1) - mean_t) @ components_t.T)[0]
    output = []
    for first in range(0, len(frames), batch_size):
        with torch.inference_mode():
            encoded = model.encode_obs({
                "visual": preprocess_frames(frames[first:first + batch_size, None], device),
                "proprio": normalise_proprio(props[first:first + batch_size, None], device)})["visual"][:, 0]
        output.append((((encoded.float().flatten(1) - mean_t) @ components_t.T) - start).cpu().numpy())
    return np.concatenate(output)


def collect(rows, cached_validation, replay, model, projection, xml, scalars, eps, device, batch_size):
    wanted = {}
    for row in rows:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    metadata, physical, actual, predicted, outcomes = [], [], [], [], []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); actions = episode.actions
            sim.reset(int(episode_id)); frames = deque([sim.render()], maxlen=3); props = deque([sim.proprio()], maxlen=3)
            for i in range(2):
                sim.execute(actions[i]); frames.append(sim.render()); props.append(sim.proprio())
            for start in range(2, len(actions) - HORIZON + 1, HORIZON):
                if start in wanted[episode_id]:
                    row = wanted[episode_id][start]; snapshot = sim.snapshot()
                    probes = offset_probe_chunks(actions[start:start + HORIZON], scalars, eps,
                                                 row.get("center_offset", 0.0))
                    is_boundary = row["stratum"] == "recentered_neighbor_boundary"
                    state, outcome, endpoint_frames, endpoint_props = simulate_responses(
                        sim, snapshot, probes, render_endpoints=is_boundary)
                    if is_boundary:
                        pred = model_probe_deltas(model, device, np.stack(frames), np.stack(props),
                                                  actions[start - 2:start], probes, projection, batch_size)
                        real = encode_real(model, device, frames[-1], props[-1], endpoint_frames,
                                           endpoint_props, projection, batch_size)
                    else:
                        index = row["validation_cache_index"]
                        real = cached_validation["actual_delta"][index]
                        pred = cached_validation["predicted_delta"][index]
                    metadata.append(dict(row)); physical.append(state); outcomes.append(outcome)
                    actual.append(real); predicted.append(pred)
                for action_index, action in enumerate(actions[start:start + HORIZON]):
                    sim.execute(action)
                    if action_index >= HORIZON - 3:
                        frames.append(sim.render()); props.append(sim.proprio())
            print(f"  collect {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return (metadata, np.stack(physical), np.stack(actual), np.stack(predicted),
            np.stack(outcomes))


def score_representation(scalars, responses, outcomes, metadata, scale, dimension=None):
    rows = []
    for meta, response, outcome in zip(metadata, responses, outcomes):
        x = response if dimension is None else response[:, :dimension]
        selected_scale = scale if dimension is None else scale[:dimension]
        result = detect_local_expansion(scalars, x, half_window=8, scale=selected_scale)
        row = dict(meta); row.update({"physical_boundary": physical_boundary(outcome, 8),
                                      "physical_counts": {"intact": int((~outcome).sum()),
                                                          "toppled": int(outcome.sum())},
                                      "alarm": result.alarm, "score": result.score,
                                      "physical_switches": int(np.sum(outcome[1:] != outcome[:-1])),
                                      "peak_count": len(result.peak_indices),
                                      "peak_scalars": list(result.peak_scalars)})
        rows.append(row)
    truth = [row["physical_boundary"] for row in rows]; alarms = [row["alarm"] for row in rows]
    boundary = [row for row in rows if row["physical_boundary"]]
    controls = [row for row in rows if row["stratum"] == "quiet_control"]
    tp = sum(row["alarm"] for row in boundary); fp = sum(row["alarm"] for row in controls)
    summary = {"confusion": confusion(truth, alarms), "controls": len(controls),
               "control_alarms": int(fp), "control_fpr": fp / max(len(controls), 1),
               "control_fpr_exact_95_percent": binomial_interval(fp, len(controls)),
               "boundaries": len(boundary), "boundary_detections": int(tp),
               "boundary_recall": tp / max(len(boundary), 1),
               "boundary_recall_exact_95_percent": binomial_interval(tp, len(boundary)),
               "score_roc_auc": float(roc_auc_score(truth, [row["score"] for row in rows])),
               "median_boundary_peak_count": float(np.median([r["peak_count"] for r in boundary])),
               "multi_switch_boundaries": int(sum(r["physical_switches"] > 1 for r in boundary))}
    return summary, rows


def write_cache(path, metadata, scalars, physical, actual, predicted, outcomes):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata_json=np.asarray(json.dumps(metadata)), scalars=scalars,
                        physical_delta=physical, actual_delta=actual,
                        predicted_delta=predicted, physical_outcomes=outcomes)


def read_cache(path):
    data = dict(np.load(path, allow_pickle=False))
    return (json.loads(str(data["metadata_json"])), data["scalars"], data["physical_delta"],
            data["actual_delta"], data["predicted_delta"], data["physical_outcomes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB)); ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--projection-cache", default=str(ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--validation-cache", default=str(ROOT / "results/jenga/neighbor_validation_cache.npz"))
    ap.add_argument("--discovery", default=str(ROOT / "results/jenga/near_boundary_discovery.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/multipeak_oracle_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/multipeak_oracle.json"))
    ap.add_argument("--controls", type=int, default=100); ap.add_argument("--boundaries", type=int, default=30)
    ap.add_argument("--probes", type=int, default=50); ap.add_argument("--eps", type=float, default=.10)
    ap.add_argument("--batch-size", type=int, default=50); ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(); scalars = probe_scalars(args.probes)
    original_meta, _, _, original_actual, _ = read_probe_cache(args.original_cache)
    image_scale = robust_component_scale(original_actual)[:4]
    physical_scale = robust_component_scale(original_physical_responses(args.original_cache))
    if args.reuse_cache:
        metadata, scalars, physical, actual, predicted, outcomes = read_cache(args.cache)
    else:
        controls, cached = choose_controls(args.validation_cache, args.controls)
        boundaries = choose_boundaries(args.discovery, args.boundaries)
        replay = JengaReplay(args.lmdb); model = load_world_model(args.checkpoint, args.device)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_multipeak_") as temp:
                metadata, physical, actual, predicted, outcomes = collect(
                    controls + boundaries, cached, replay, model,
                    projection_basis(args.projection_cache), extract_sim(args.sim_archive, temp),
                    scalars, args.eps, args.device, args.batch_size)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, physical, actual, predicted, outcomes)
    representations = {}
    for name, response, scale, dimension in (
            ("physical_oracle", physical, physical_scale, None),
            ("real_dino", actual, image_scale, 4),
            ("predicted_dino_wm", predicted, image_scale, 4)):
        if name == "physical_oracle" and response.shape[-1] != len(scale):
            response = np.concatenate(
                [response[:, :, :18], response[:, :, 18 + NEIGHBOR_CONTACT_INDICES]], axis=-1)
        summary, rows = score_representation(scalars, response, outcomes, metadata, scale, dimension)
        representations[name] = {"summary": summary, "rows": rows}
    oracle = representations["physical_oracle"]["summary"]
    real = representations["real_dino"]["summary"]
    pred = representations["predicted_dino_wm"]["summary"]
    decision = {
        "physical_oracle_passed": bool(oracle["control_fpr"] <= .05 and oracle["boundary_recall"] >= .8),
        "real_encoding_passed": bool(real["control_fpr"] <= .05 and real["boundary_recall"] >= .8),
        "world_model_passed": bool(pred["control_fpr"] <= .05 and pred["boundary_recall"] >= .8),
    }
    if not decision["physical_oracle_passed"]:
        next_step = "revise the stability score; representation or world-model retraining is not justified"
    elif not decision["real_encoding_passed"]:
        next_step = "improve the self-supervised state representation before retraining dynamics"
    elif not decision["world_model_passed"]:
        next_step = "train a stochastic branch-preserving counterfactual world model"
    else:
        next_step = "optimize and test the monitor online"
    result = {"protocol": {"controls": args.controls, "recentered_boundaries": args.boundaries,
                           "probes": args.probes, "eps": args.eps, "horizon": HORIZON,
                           "held_tail": TAIL, "score": "local piecewise-linear continuity BIC; positive evidence",
                           "physical_scale_source": f"original {len(original_meta)}-chunk physical cache",
                           "image_scale_source": f"original {len(original_meta)}-chunk real DINO cache"},
              "representations": representations, "decision": decision,
              "recommended_next_step": next_step}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": result["protocol"],
                      "summaries": {k: v["summary"] for k, v in representations.items()},
                      "decision": decision, "recommended_next_step": next_step}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
