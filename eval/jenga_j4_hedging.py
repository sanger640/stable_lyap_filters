"""PLAN_JENGA J4: locate where predicted intact/toppled geometry is lost.

This is a diagnostic, not a fitted monitor. Outcome labels are used only to score geometry after
the unsupervised residualisation/PCA/HDBSCAN pipeline has run. The J2/J3 cache supplies long-tail
predictions; this script adds paired encoded truth at those checkpoints and short predictions at
horizons 1, 2, 4, and 8.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basins import ProprioResidualizer, fit_basin_model             # noqa: E402
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, NUM_HIST, JengaReplay,
                           load_world_model, normalise_actions, normalise_proprio,
                           preprocess_frames)                       # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_j2_j3_predicted_basins import best_agreement, separation_ratio  # noqa: E402

EARLY_HORIZONS = (1, 2, 4, 8)


def outcome_axis(points, truth):
    """RMS distance between outcome centroids and its unit direction."""
    x = np.asarray(points, np.float32)
    y = np.asarray(truth, int)
    delta = x[y == 1].mean(0) - x[y == 0].mean(0)
    rms = float(np.sqrt(np.mean(delta ** 2)))
    norm = float(np.linalg.norm(delta))
    direction = delta / max(norm, 1e-12)
    return rms, direction


def axis_cosine(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def nearest_centroid_accuracy(train, query, truth):
    """Classify query points with the two labelled train centroids (diagnostic only)."""
    a = np.asarray(train, np.float32); q = np.asarray(query, np.float32)
    y = np.asarray(truth, int)
    centers = np.stack([a[y == c].mean(0) for c in (0, 1)])
    pred = ((q[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    return float((pred == y).mean()), [int((pred == c).sum()) for c in (0, 1)]


def pca_projection(points, dimension):
    x = np.asarray(points, np.float32)
    centered = x - x.mean(0)
    components = np.linalg.svd(centered, full_matrices=False)[2][:dimension]
    return centered @ components.T


def cluster_summary(points, truth, pca_dim, min_fraction):
    """Unsupervised cluster result, with labels consulted only for post-fit scoring."""
    x = pca_projection(points, pca_dim)
    try:
        basin = fit_basin_model(x, pca_dim, min_fraction)
    except ValueError as ex:
        return {"n_clusters": 0, "agreement": 0.0, "coverage": 0.0,
                "cluster_sizes": [], "cluster_composition": [], "error": str(ex)}
    agreement, coverage = best_agreement(basin.labels, truth)
    composition = []
    for c in range(basin.n_clusters):
        member = basin.labels == c
        composition.append({"cluster": c, "intact": int(((truth == 0) & member).sum()),
                            "toppled": int(((truth == 1) & member).sum())})
    noise = basin.labels < 0
    return {
        "n_clusters": basin.n_clusters,
        "agreement": agreement,
        "coverage": coverage,
        "cluster_sizes": sorted(basin.cluster_sizes, reverse=True),
        "cluster_composition": composition,
        "noise_composition": {"intact": int(((truth == 0) & noise).sum()),
                              "toppled": int(((truth == 1) & noise).sum())},
    }


def paired_geometry(predicted, actual, proprio, truth, pca_dim, min_fractions):
    """Compare task geometry in paired predicted and encoded-real endpoint latents."""
    pred = np.asarray(predicted, np.float32).reshape(len(predicted), -1)
    real = np.asarray(actual, np.float32).reshape(len(actual), -1)
    prop = np.asarray(proprio, np.float32); truth = np.asarray(truth, int)
    real_fit = ProprioResidualizer.fit(real, prop)
    pred_fit = ProprioResidualizer.fit(pred, prop)
    real_res = real_fit.transform(real, prop)
    pred_res = pred_fit.transform(pred, prop)
    # Apply the real residualizer to both for a paired error and true-centroid diagnostic.
    pred_in_real_frame = real_fit.transform(pred, prop)
    error = pred_in_real_frame - real_res
    rmse = float(np.sqrt(np.mean(error ** 2)))
    nrmse = float(rmse / max(float(real_res.std()), 1e-12))
    real_rms, real_axis = outcome_axis(real_res, truth)
    pred_rms, pred_axis = outcome_axis(pred_res, truth)
    real_pca = pca_projection(real_res, pca_dim)
    pred_pca = pca_projection(pred_res, pca_dim)
    centroid_accuracy, centroid_counts = nearest_centroid_accuracy(
        real_res, pred_in_real_frame, truth)
    return {
        "prediction_rmse": rmse,
        "prediction_normalised_rmse": nrmse,
        "real_outcome_centroid_rms": real_rms,
        "predicted_outcome_centroid_rms": pred_rms,
        "outcome_axis_contraction": float(pred_rms / max(real_rms, 1e-12)),
        "outcome_axis_cosine": axis_cosine(real_axis, pred_axis),
        "error_to_real_outcome_axis": float(rmse / max(real_rms, 1e-12)),
        "real_separation_ratio": separation_ratio(real_pca, truth),
        "predicted_separation_ratio": separation_ratio(pred_pca, truth),
        "predicted_to_real_centroid_accuracy": centroid_accuracy,
        "predicted_to_real_centroid_counts": {"intact": centroid_counts[0],
                                              "toppled": centroid_counts[1]},
        "hdbscan": {
            f"min{f:g}": {
                "real": cluster_summary(real_res, truth, pca_dim, f),
                "predicted": cluster_summary(pred_res, truth, pca_dim, f),
            } for f in min_fractions
        },
    }


def collect_short_and_truth(args, replay, ids, model, long_cache):
    early_pred, early_truth, early_prop, tail_truth = [], [], [], []
    early_indices = [NUM_HIST + h - 1 for h in EARLY_HORIZONS]
    for n, ep_id in enumerate(ids):
        ep = replay.episode(ep_id)
        tail_indices = [int(x) for x in long_cache["target_indices"][n]]
        target_indices = sorted(set(early_indices + tail_indices))
        frame_indices = list(range(NUM_HIST)) + target_indices
        frames = replay.frames(ep, frame_indices)
        visual = preprocess_frames(frames, args.device)
        prop = normalise_proprio(ep.proprio[frame_indices], args.device).unsqueeze(0)
        actions = normalise_actions(ep.actions[:early_indices[-1]], args.device).unsqueeze(0)
        with torch.inference_mode():
            encoded = model.encode_obs({"visual": visual, "proprio": prop})["visual"][0]
            pred, _ = model.rollout(
                {"visual": visual[:, :NUM_HIST], "proprio": prop[:, :NUM_HIST]}, actions)
        pred = pred["visual"][0]
        if pred.shape[0] <= early_indices[-1] or not torch.isfinite(pred).all():
            raise RuntimeError(f"episode {ep_id}: invalid short rollout shaped {tuple(pred.shape)}")
        encoded_by_index = {idx: encoded[NUM_HIST + j].float().cpu().numpy()
                            for j, idx in enumerate(target_indices)}
        early_pred.append(np.stack([pred[i].float().cpu().numpy() for i in early_indices]))
        early_truth.append(np.stack([encoded_by_index[i] for i in early_indices]))
        early_prop.append(ep.proprio[early_indices])
        tail_truth.append(np.stack([encoded_by_index[i] for i in tail_indices]))
        print(f"  {n + 1}/{len(ids)} ep{ep_id}", flush=True)
    return {
        "episode_ids": np.asarray(ids),
        "early_horizons": np.asarray(EARLY_HORIZONS, np.int32),
        "early_predicted": np.asarray(early_pred, np.float16),
        "early_truth": np.asarray(early_truth, np.float16),
        "early_proprio": np.asarray(early_prop, np.float32),
        "tail_truth": np.asarray(tail_truth, np.float16),
    }


def analyse(short, long_cache, labels, pca_dim, min_fractions):
    ids = [str(x) for x in long_cache["episode_ids"]]
    truth = np.array([labels[e]["outcome"] != "success" for e in ids], int)
    rows = []
    for j, h in enumerate(short["early_horizons"]):
        metric = paired_geometry(short["early_predicted"][:, j], short["early_truth"][:, j],
                                 short["early_proprio"][:, j], truth, pca_dim, min_fractions)
        metric.update({"stage": "short_horizon", "horizon": int(h)})
        rows.append(metric)
    for j, fraction in enumerate(long_cache["tail_fractions"]):
        metric = paired_geometry(long_cache["endings"][:, j], short["tail_truth"][:, j],
                                 long_cache["proprio"][:, j], truth, pca_dim, min_fractions)
        metric.update({"stage": "nominal_tail", "tail_fraction": float(fraction),
                       "median_steps_after_chunk": float(np.median(
                           long_cache["target_indices"][:, j] - long_cache["target_indices"][:, 0]))})
        rows.append(metric)
    return {"episodes": len(ids), "truth_counts": {"intact": int((truth == 0).sum()),
                                                     "toppled": int((truth == 1).sum())},
            "pca_dim": pca_dim, "min_cluster_fractions": list(min_fractions),
            "checkpoints": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--labels", default=str(ROOT / "data/jenga/labels_noise100.json"))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--j2-cache", default=str(ROOT / "results/jenga/j2_j3_predicted_endings.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/j4_hedging.json"))
    ap.add_argument("--pca", type=int, default=2)
    ap.add_argument("--min-cluster-fractions", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    long_cache = dict(np.load(args.j2_cache, allow_pickle=False))
    ids = [str(x) for x in long_cache["episode_ids"]]
    cache_path = Path(args.cache)
    replay = JengaReplay(args.lmdb)
    try:
        if args.reuse_cache and cache_path.exists():
            short = dict(np.load(cache_path, allow_pickle=False))
            if list(short["episode_ids"].astype(str)) != ids:
                raise ValueError("J4 cache episode ids do not match J2 cache")
            print(f"loaded {cache_path}")
        else:
            model = load_world_model(args.checkpoint, args.device)
            print(f"collecting paired truth and horizons {EARLY_HORIZONS} for {len(ids)} episodes")
            short = collect_short_and_truth(args, replay, ids, model, long_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, **short)
            print(f"cached -> {cache_path}")
    finally:
        replay.close()

    result = analyse(short, long_cache, json.load(open(args.labels)), args.pca,
                     args.min_cluster_fractions)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()),
                   "lmdb": str(Path(args.lmdb).resolve())})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("\nJ4 horizon diagnostic")
    print("stage             point   nRMSE  contraction axis-cos pred-sep real-sep  k(5%) agree")
    for row in result["checkpoints"]:
        point = (f"H={row['horizon']}" if row["stage"] == "short_horizon"
                 else f"tail={row['tail_fraction']:.2f}")
        hdb = row["hdbscan"]["min0.05"]["predicted"]
        print(f"{row['stage']:<18}{point:<9}{row['prediction_normalised_rmse']:>7.3f}"
              f"{row['outcome_axis_contraction']:>12.3f}{row['outcome_axis_cosine']:>9.3f}"
              f"{row['predicted_separation_ratio']:>9.2f}{row['real_separation_ratio']:>9.2f}"
              f"{hdb['n_clusters']:>7}{hdb['agreement']:>7.1%}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
