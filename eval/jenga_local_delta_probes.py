"""Local delta-z probe experiment for the chunk-level Jenga monitor.

Each selected simulator state receives 50 coherent scalar perturbations of its H=8 action chunk,
followed by the selected five-step held tail. The same branches run through MuJoCo and DINO-WM.
Clustering uses endpoint change relative to the shared start, after regressing out a smooth
quadratic response to probe magnitude. Labels are used only to evaluate the unsupervised split.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basins import dissent_count, fit_basin_model, known_coverage  # noqa: E402
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay, load_world_model,
                           normalise_actions, normalise_proprio, preprocess_frames)  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_j2_j3_predicted_basins import best_agreement, separation_ratio  # noqa: E402
from jenga_short_held_tails import (DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim,
                                    BLOCK_NAMES, CONTACT_LABELS, projection_basis)  # noqa: E402

TAIL = 5


def probe_scalars(n):
    """Deterministic Gaussian quantiles: Gaussian sizing without Monte Carlo imbalance."""
    return norm.ppf((np.arange(int(n), dtype=np.float64) + 0.5) / int(n)).astype(np.float32)


def coherent_probe_chunks(chunk, scalars, eps):
    chunk = np.asarray(chunk, np.float32)
    z = np.asarray(scalars, np.float32)
    direction = chunk - chunk[:1]
    direction[:, 3] = 0.0
    probes = chunk[None] + float(eps) * z[:, None, None] * direction[None]
    probes[:, :, 3] = chunk[None, :, 3]
    return probes


def remove_smooth_probe_response(delta, scalars):
    """Remove constant/linear/quadratic action response; retain branch discontinuities."""
    x = np.asarray(delta, np.float32)
    z = np.asarray(scalars, np.float32)
    design = np.stack([np.ones(len(z)), z, z ** 2], axis=1)
    return x - design @ np.linalg.lstsq(design, x, rcond=None)[0]


def local_clusters(delta, scalars, min_fraction):
    from sklearn.decomposition import PCA
    residual = remove_smooth_probe_response(delta, scalars)
    x = PCA(n_components=2, svd_solver="full").fit_transform(residual)
    try:
        basin = fit_basin_model(x, 2, min_fraction)
    except ValueError as ex:
        return np.full(len(x), -1, int), {
            "n_clusters": 0, "dissent": 0, "coverage": 0.0,
            "cluster_sizes": [], "error": str(ex),
        }
    return basin.labels, {
        "n_clusters": basin.n_clusters,
        "dissent": dissent_count(basin.labels, basin.n_clusters),
        "coverage": known_coverage(basin.labels, basin.n_clusters),
        "cluster_sizes": sorted(basin.cluster_sizes, reverse=True),
    }


def choose_chunks(short_cache, near_count, far_count):
    """Select event-adjacent chunks plus quiet controls; selection never uses model output."""
    data = dict(np.load(short_cache, allow_pickle=False))
    ids = data["episode_ids"].astype(str)
    starts = data["chunk_starts"].astype(int)
    peak0 = np.asarray(data["tail0_peak_tilt"], np.float32)
    peak5 = np.asarray(data["tail5_peak_tilt"], np.float32)
    new5 = np.asarray(data["tail5_new_topple"], bool)
    start_tilt = np.zeros(len(ids), np.float32)
    previous_episode, previous_tilt = None, 0.0
    for i, episode in enumerate(ids):
        if episode != previous_episode:
            previous_tilt = 0.0
        start_tilt[i] = previous_tilt
        previous_tilt = float(data["tail0_end_tilt"][i])
        previous_episode = episode

    # New topples first, then largest tail-induced motion while starting upright.
    eligible = np.where(start_tilt < TOPPLE_DEG)[0]
    near_order = sorted(eligible, key=lambda i: (not new5[i], -(peak5[i] - start_tilt[i]),
                                                 abs(peak5[i] - TOPPLE_DEG), int(ids[i]), starts[i]))
    near = near_order[:min(int(near_count), len(near_order))]
    quiet = [i for i in eligible if peak5[i] < 5.0 and i not in set(near)]
    # Spread controls through the corpus rather than taking one episode prefix.
    if quiet:
        positions = np.linspace(0, len(quiet) - 1, min(int(far_count), len(quiet))).round().astype(int)
        far = [quiet[i] for i in positions]
    else:
        far = []
    rows = []
    for stratum, selected in (("event_adjacent", near), ("quiet_control", far)):
        for i in selected:
            rows.append({"episode_id": ids[i], "chunk_start": int(starts[i]),
                         "stratum": stratum, "screen_start_tilt": float(start_tilt[i]),
                         "screen_peak_tail5": float(peak5[i]),
                         "screen_new_topple_tail5": bool(new5[i])})
    return rows


def model_probe_deltas(model, device, history_frames, history_proprio, history_actions,
                       probe_chunks, projection, batch_size):
    mean, components = projection
    mean_t = torch.as_tensor(mean, device=device)
    components_t = torch.as_tensor(components, device=device)
    initial_visual = preprocess_frames(history_frames[None], device)
    initial_prop = normalise_proprio(history_proprio[None], device)
    with torch.inference_mode():
        encoded_start = model.encode_obs({"visual": initial_visual[:, -1:],
                                          "proprio": initial_prop[:, -1:]})["visual"][:, 0]
    start_projected = ((encoded_start.float().flatten(1) - mean_t) @ components_t.T)[0]
    outputs = []
    for first in range(0, len(probe_chunks), batch_size):
        probes = probe_chunks[first:first + batch_size]
        n = len(probes)
        held = np.repeat(probes[:, -1:], TAIL, axis=1)
        history = np.repeat(np.asarray(history_actions, np.float32)[None], n, axis=0)
        actions = np.concatenate([history, probes, held], axis=1)
        obs = {"visual": initial_visual.repeat(n, 1, 1, 1, 1),
               "proprio": initial_prop.repeat(n, 1, 1)}
        with torch.inference_mode():
            prediction, _ = model.rollout(obs, normalise_actions(actions, device))
        endpoint = prediction["visual"][:, 10 + TAIL].float().flatten(1)
        outputs.append(((endpoint - mean_t) @ components_t.T - start_projected).cpu().numpy())
    return np.concatenate(outputs)


def true_probe_deltas(model, device, sim, start_snapshot, start_frame, start_prop,
                      probe_chunks, projection, batch_size, collect_diagnostics=True):
    endpoint_frames, endpoint_props, peak_tilts = [], [], []
    start_blocks = sim.block_diagnostics() if collect_diagnostics else None
    start_contact = sim.contact_signature() if collect_diagnostics else None
    start_ee = sim.data.site_xpos[sim.site].copy() if collect_diagnostics else None
    diagnostics = {key: [] for key in (
        "peak_tilt", "end_tilt", "block_end_position", "block_end_quaternion",
        "block_displacement", "block_rotation_deg", "contact_end", "contact_seen",
        "contact_transitions", "ee_end_position", "ee_displacement")}
    for probe in probe_chunks:
        sim.restore(start_snapshot)
        peak = sim.max_tilt()
        trace = None
        if collect_diagnostics:
            trace = {"peak_tilt": start_blocks["tilt"].copy(),
                     "contact_seen": start_contact.copy(),
                     "contact_transitions": np.zeros(len(CONTACT_LABELS), np.int32),
                     "previous_contact": start_contact.copy()}
        for action in probe:
            peak = max(peak, sim.execute(action, trace))
        for _ in range(TAIL):
            peak = max(peak, sim.execute(probe[-1], trace))
        if collect_diagnostics:
            end_blocks = sim.block_diagnostics(); end_contact = sim.contact_signature()
            quaternion_dot = np.abs(np.sum(
                start_blocks["quaternion"] * end_blocks["quaternion"], axis=1))
            rotation = np.degrees(2 * np.arccos(np.clip(quaternion_dot, 0, 1)))
            end_ee = sim.data.site_xpos[sim.site].copy()
            diagnostics["peak_tilt"].append(trace["peak_tilt"])
            diagnostics["end_tilt"].append(end_blocks["tilt"])
            diagnostics["block_end_position"].append(end_blocks["position"])
            diagnostics["block_end_quaternion"].append(end_blocks["quaternion"])
            diagnostics["block_displacement"].append(np.linalg.norm(
                end_blocks["position"] - start_blocks["position"], axis=1))
            diagnostics["block_rotation_deg"].append(rotation)
            diagnostics["contact_end"].append(end_contact)
            diagnostics["contact_seen"].append(trace["contact_seen"])
            diagnostics["contact_transitions"].append(trace["contact_transitions"])
            diagnostics["ee_end_position"].append(end_ee)
            diagnostics["ee_displacement"].append(np.linalg.norm(end_ee - start_ee))
        endpoint_frames.append(sim.render()); endpoint_props.append(sim.proprio())
        peak_tilts.append(peak)
    mean, components = projection
    mean_t = torch.as_tensor(mean, device=device)
    components_t = torch.as_tensor(components, device=device)
    with torch.inference_mode():
        start = model.encode_obs({
            "visual": preprocess_frames(start_frame[None, None], device),
            "proprio": normalise_proprio(start_prop[None, None], device)})["visual"][:, 0]
    start_projected = ((start.float().flatten(1) - mean_t) @ components_t.T)[0]
    projected = []
    for first in range(0, len(endpoint_frames), batch_size):
        frames = np.stack(endpoint_frames[first:first + batch_size])[:, None]
        props = np.stack(endpoint_props[first:first + batch_size])[:, None]
        with torch.inference_mode():
            encoded = model.encode_obs({"visual": preprocess_frames(frames, device),
                                        "proprio": normalise_proprio(props, device)})["visual"][:, 0]
        projected.append(((encoded.float().flatten(1) - mean_t) @ components_t.T
                          - start_projected).cpu().numpy())
    sim.restore(start_snapshot)
    diagnostics = ({key: np.asarray(value) for key, value in diagnostics.items()}
                   if collect_diagnostics else {})
    if collect_diagnostics:
        diagnostics["start_block_position"] = start_blocks["position"]
        diagnostics["start_block_quaternion"] = start_blocks["quaternion"]
        diagnostics["start_contact"] = start_contact
        diagnostics["start_ee_position"] = start_ee
    return np.concatenate(projected), np.asarray(peak_tilts, np.float32), diagnostics


def score_chunk(pred_delta, true_delta, physical, scalars, min_fraction):
    pred_labels, pred_summary = local_clusters(pred_delta, scalars, min_fraction)
    true_labels, true_summary = local_clusters(true_delta, scalars, min_fraction)
    physical = np.asarray(physical, int)
    counts = np.bincount(physical, minlength=2)
    physical_dissent = int(min(counts))
    physical_boundary = physical_dissent >= 2
    pred_agreement, pred_label_coverage = best_agreement(pred_labels, physical)
    true_agreement, true_label_coverage = best_agreement(true_labels, physical)
    true_pca = remove_smooth_probe_response(true_delta, scalars)
    pred_pca = remove_smooth_probe_response(pred_delta, scalars)
    has_both = bool((counts > 0).all())
    return {
        "physical_counts": {"intact": int(counts[0]), "toppled": int(counts[1])},
        "physical_dissent": physical_dissent, "physical_boundary": bool(physical_boundary),
        "predicted": pred_summary, "actual_latent": true_summary,
        "predicted_alarm": bool(pred_summary["dissent"] >= 2),
        "actual_latent_alarm": bool(true_summary["dissent"] >= 2),
        "predicted_cluster_physical_agreement": pred_agreement if np.isfinite(pred_agreement) else None,
        "predicted_cluster_physical_coverage": pred_label_coverage,
        "actual_cluster_physical_agreement": true_agreement if np.isfinite(true_agreement) else None,
        "actual_cluster_physical_coverage": true_label_coverage,
        "predicted_physical_separation": separation_ratio(pred_pca, physical) if has_both else None,
        "actual_physical_separation": separation_ratio(true_pca, physical) if has_both else None,
    }


def confusion(rows, key):
    truth = np.array([r["physical_boundary"] for r in rows], bool)
    pred = np.array([r[key] for r in rows], bool)
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def write_probe_cache(path, rows, scalars, predicted_delta, actual_delta, physical,
                      diagnostics=None):
    """Persist the expensive paired branches for detector re-analysis."""
    metadata_keys = ("episode_id", "chunk_start", "stratum", "screen_start_tilt",
                     "screen_peak_tail5", "screen_new_topple_tail5", "start_tilt")
    metadata = [{key: row[key] for key in metadata_keys} for row in rows]
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"scalars": np.asarray(scalars, np.float32),
              "predicted_delta": np.asarray(predicted_delta, np.float32),
              "actual_delta": np.asarray(actual_delta, np.float32),
              "physical": np.asarray(physical, bool),
              "metadata_json": np.asarray(json.dumps(metadata)),
              "block_names": np.asarray(BLOCK_NAMES),
              "contact_labels": np.asarray(CONTACT_LABELS)}
    if diagnostics:
        arrays.update({f"physical_{key}": np.asarray(value)
                       for key, value in diagnostics.items()})
    np.savez_compressed(output, **arrays)


def read_probe_cache(path):
    data = dict(np.load(path, allow_pickle=False))
    metadata = json.loads(str(data["metadata_json"]))
    return metadata, data["scalars"], data["predicted_delta"], data["actual_delta"], data["physical"]


def read_probe_cache_with_diagnostics(path):
    data = dict(np.load(path, allow_pickle=False))
    base = read_probe_cache(path)
    diagnostics = {key.removeprefix("physical_"): value for key, value in data.items()
                   if key.startswith("physical_")}
    diagnostics["block_names"] = data.get("block_names", np.asarray(BLOCK_NAMES))
    diagnostics["contact_labels"] = data.get("contact_labels", np.asarray(CONTACT_LABELS))
    return (*base, diagnostics)


def rescore_cached(metadata, scalars, predicted_delta, actual_delta, physical, min_fraction):
    rows = []
    for meta, pred, actual, outcome in zip(metadata, predicted_delta, actual_delta, physical):
        row = score_chunk(pred, actual, outcome, scalars, min_fraction)
        row.update(meta); rows.append(row)
    return rows


def run(args, selected, replay, model, projection, xml):
    selected_by_episode = {}
    for row in selected:
        selected_by_episode.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    scalars = probe_scalars(args.probes)
    sim = DirectJengaSim(xml); output = []
    predicted_delta, actual_delta, physical_outcomes = [], [], []
    all_diagnostics = {}
    try:
        for episode_number, ep_id in enumerate(sorted(selected_by_episode, key=int)):
            episode = replay.episode(ep_id); actions = episode.actions
            sim.reset(int(ep_id))
            frames = deque([sim.render()], maxlen=3); props = deque([sim.proprio()], maxlen=3)
            for i in range(2):
                sim.execute(actions[i]); frames.append(sim.render()); props.append(sim.proprio())
            wanted = selected_by_episode[ep_id]
            for start in range(2, len(actions) - HORIZON + 1, HORIZON):
                if start in wanted:
                    snapshot = sim.snapshot(); start_frame = frames[-1]; start_prop = props[-1]
                    probes = coherent_probe_chunks(actions[start:start + HORIZON], scalars, args.eps)
                    pred_delta = model_probe_deltas(
                        model, args.device, np.stack(frames), np.stack(props), actions[start - 2:start],
                        probes, projection, args.batch_size)
                    true_delta, peak, diagnostics = true_probe_deltas(
                        model, args.device, sim, snapshot, start_frame, start_prop, probes,
                        projection, args.batch_size)
                    physical = peak >= TOPPLE_DEG
                    scored = score_chunk(pred_delta, true_delta, physical, scalars,
                                         args.min_cluster_fraction)
                    scored.update(wanted[start]); scored["start_tilt"] = float(sim.max_tilt())
                    output.append(scored)
                    predicted_delta.append(pred_delta); actual_delta.append(true_delta)
                    physical_outcomes.append(physical)
                    for key, value in diagnostics.items():
                        all_diagnostics.setdefault(key, []).append(value)
                # Restore happened inside true_probe_deltas; advance the unperturbed trajectory.
                for action_index, action in enumerate(actions[start:start + HORIZON]):
                    sim.execute(action)
                    if action_index >= HORIZON - 3:
                        frames.append(sim.render()); props.append(sim.proprio())
            print(f"  {episode_number + 1}/{len(selected_by_episode)} ep{ep_id}", flush=True)
    finally:
        sim.close()
    return (output, np.stack(predicted_delta), np.stack(actual_delta),
            np.stack(physical_outcomes),
            {key: np.stack(value) for key, value in all_diagnostics.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--short-cache", default=str(ROOT / "results/jenga/short_held_tails_cache.npz"))
    ap.add_argument("--projection-cache", default=str(
        ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--near-chunks", type=int, default=50)
    ap.add_argument("--far-chunks", type=int, default=50)
    ap.add_argument("--probes", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--min-cluster-fraction", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default=str(
        ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--output", default=str(ROOT / "results/jenga/local_delta_probes.json"))
    args = ap.parse_args()

    if args.reuse_cache:
        metadata, scalars, predicted, actual, physical = read_probe_cache(args.cache)
        rows = rescore_cached(metadata, scalars, predicted, actual, physical,
                              args.min_cluster_fraction)
    else:
        selected = choose_chunks(args.short_cache, args.near_chunks, args.far_chunks)
        replay = JengaReplay(args.lmdb); model = load_world_model(args.checkpoint, args.device)
        projection = projection_basis(args.projection_cache)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_delta_") as temp:
                rows, predicted, actual, physical, diagnostics = run(
                    args, selected, replay, model, projection,
                    extract_sim(args.sim_archive, temp))
        finally:
            replay.close()
        scalars = probe_scalars(args.probes)
        write_probe_cache(args.cache, rows, scalars, predicted, actual, physical, diagnostics)
    result = {
        "episodes_replayed": len(set(r["episode_id"] for r in rows)), "chunks": len(rows),
        "horizon": HORIZON, "held_tail": TAIL, "probes": args.probes, "eps": args.eps,
        "probe_scalars": {"min": float(np.min(scalars)),
                          "max": float(np.max(scalars))},
        "selection": {"event_adjacent": sum(r["stratum"] == "event_adjacent" for r in rows),
                      "quiet_control": sum(r["stratum"] == "quiet_control" for r in rows)},
        "physical_boundary_chunks": sum(r["physical_boundary"] for r in rows),
        "predicted_confusion": confusion(rows, "predicted_alarm"),
        "actual_latent_confusion": confusion(rows, "actual_latent_alarm"),
        "rows": rows,
        "note": "stratified diagnostic set, so precision is not a deployment-prevalence estimate",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("\nlocal delta-z result")
    print(f"{len(rows)} chunks; {result['physical_boundary_chunks']} physical probe boundaries")
    print("predicted", result["predicted_confusion"])
    print("actual latent", result["actual_latent_confusion"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
