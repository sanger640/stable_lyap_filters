"""PLAN_JENGA J2/J3: nominal-continuation tail and predicted-ending basins.

J2 uses the recorded remainder of each policy trajectory as an in-distribution settle tail.  A
single full rollout supplies checkpoints after 0%, 25%, 50%, 75%, 90%, and 100% of that tail.  J3
regresses raw arm proprioception out of the predicted visual endings, applies PCA, and lets HDBSCAN
discover the basin count.  Outcome labels are used only after fitting for evaluation.
"""
import argparse
from itertools import permutations
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

DEFAULT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def checkpoint_indices(length, horizon, fractions=DEFAULT_FRACTIONS):
    """Frame indices after H steps and increasing fractions of nominal continuation."""
    final = int(length) - 1
    chunk_end = NUM_HIST + int(horizon) - 1
    if chunk_end >= final:
        raise ValueError(f"episode length {length} is too short for H={horizon}")
    return np.array([chunk_end + round(float(f) * (final - chunk_end)) for f in fractions], int)


def best_agreement(labels, truth):
    """Permutation-invariant agreement over non-noise labels, plus their coverage."""
    labels = np.asarray(labels, int); truth = np.asarray(truth, int)
    keep = labels >= 0
    if not keep.any():
        return 0.0, 0.0
    lab, target = labels[keep], truth[keep]
    ks = int(lab.max()) + 1
    kt = int(target.max()) + 1
    if ks > 7:
        return float("nan"), float(keep.mean())
    score = max(float((np.asarray(p)[lab] == target).mean())
                for p in permutations(range(max(ks, kt)), ks))
    return score, float(keep.mean())


def separation_ratio(points, truth):
    """Median between-class distance divided by median within-class distance."""
    x = np.asarray(points, np.float32); y = np.asarray(truth, int)
    d = np.sqrt(np.maximum(
        (x * x).sum(1)[:, None] + (x * x).sum(1)[None] - 2 * x @ x.T, 0.0))
    iu = np.triu_indices(len(x), 1)
    same = y[iu[0]] == y[iu[1]]
    within, between = d[iu][same], d[iu][~same]
    return float(np.median(between) / max(float(np.median(within)), 1e-12))


def label_stability(previous, current):
    """Permutation-invariant agreement where both independently fitted models are non-noise."""
    a = np.asarray(previous, int); b = np.asarray(current, int)
    keep = (a >= 0) & (b >= 0)
    if not keep.any():
        return 0.0, 0.0
    score, _ = best_agreement(a[keep], b[keep])
    return score, float(keep.mean())


def portable(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect(args, replay, ids, model):
    fractions = tuple(args.tail_fractions)
    endpoints, endpoint_proprio, targets = [], [], []
    one_step_rmse, one_step_nrmse = [], []
    for n, ep_id in enumerate(ids):
        ep = replay.episode(ep_id)
        idx = checkpoint_indices(ep.length, args.horizon, fractions)
        # Four real frames provide the three-frame warm start and the one-step truth target.
        frames = replay.frames(ep, range(NUM_HIST + 1))
        visual = preprocess_frames(frames, args.device)
        prop4 = normalise_proprio(ep.proprio[:NUM_HIST + 1], args.device).unsqueeze(0)
        actions = normalise_actions(ep.actions[:ep.length - 1], args.device).unsqueeze(0)
        with torch.inference_mode():
            pred, _ = model.rollout(
                {"visual": visual[:, :NUM_HIST], "proprio": prop4[:, :NUM_HIST]}, actions)
            truth1 = model.encode_obs(
                {"visual": visual[:, NUM_HIST:NUM_HIST + 1],
                 "proprio": prop4[:, NUM_HIST:NUM_HIST + 1]})["visual"][:, 0]
        if pred["visual"].shape[1] != ep.length:
            raise RuntimeError(
                f"episode {ep_id}: rollout length {pred['visual'].shape[1]} != {ep.length}")
        if not torch.isfinite(pred["visual"]).all():
            raise RuntimeError(f"episode {ep_id}: rollout produced a non-finite visual latent")
        p1 = pred["visual"][:, NUM_HIST]
        rmse = torch.sqrt(((p1.float() - truth1.float()) ** 2).mean())
        one_step_rmse.append(float(rmse))
        one_step_nrmse.append(float(rmse / truth1.float().std().clamp_min(1e-8)))
        endpoints.append(pred["visual"][0, idx].float().cpu().numpy().reshape(len(idx), -1))
        endpoint_proprio.append(ep.proprio[idx])
        targets.append(idx)
        print(f"  {n + 1}/{len(ids)} ep{ep_id}: length={ep.length}, "
              f"tail={idx[-1] - idx[0]} steps", flush=True)
    return {
        "endings": np.asarray(endpoints, np.float16),
        "proprio": np.asarray(endpoint_proprio, np.float32),
        "target_indices": np.asarray(targets, np.int32),
        "one_step_rmse": np.asarray(one_step_rmse, np.float32),
        "one_step_nrmse": np.asarray(one_step_nrmse, np.float32),
        "episode_ids": np.asarray(ids),
        "tail_fractions": np.asarray(fractions, np.float32),
    }


def analyse(data, labels, pca_dims, min_fractions):
    ids = [str(x) for x in data["episode_ids"]]
    truth = np.array([labels[e]["outcome"] != "success" for e in ids], int)
    tilts = np.array([labels[e]["peak_tilt_deg"] for e in ids], np.float32)
    fractions = [float(x) for x in data["tail_fractions"]]
    results = {}
    fitted = {}
    residuals, projected = [], []
    max_pca = max(pca_dims)
    for j in range(len(fractions)):
        e = np.asarray(data["endings"][:, j], np.float32)
        residualizer = ProprioResidualizer.fit(e, data["proprio"][:, j])
        residual = residualizer.transform(e, data["proprio"][:, j])
        centered = residual - residual.mean(0)
        components = np.linalg.svd(centered, full_matrices=False)[2][:max_pca]
        residuals.append(residual)
        projected.append(centered @ components.T)
    for pca in pca_dims:
        for min_fraction in min_fractions:
            key = f"pca{pca}_min{min_fraction:g}"
            rows, models = [], []
            for j, fraction in enumerate(fractions):
                try:
                    x = projected[j][:, :pca]
                    basin = fit_basin_model(x, pca, min_fraction)
                    agreement, coverage = best_agreement(basin.labels, truth)
                    sep = separation_ratio(basin.transform(x), truth)
                    row = {
                        "tail_fraction": fraction,
                        "n_clusters": basin.n_clusters,
                        "agreement": agreement,
                        "coverage": coverage,
                        "separation_ratio": sep,
                        "cluster_sizes": sorted(basin.cluster_sizes, reverse=True),
                    }
                    models.append(basin)
                except ValueError as ex:
                    row = {"tail_fraction": fraction, "n_clusters": 0, "agreement": 0.0,
                           "coverage": 0.0, "separation_ratio": None,
                           "cluster_sizes": [], "error": str(ex)}
                    models.append(None)
                rows.append(row)
            stability = []
            for a, b in zip(models, models[1:]):
                if a is None or b is None:
                    stability.append({"agreement": 0.0, "joint_coverage": 0.0})
                else:
                    score, coverage = label_stability(a.labels, b.labels)
                    stability.append({"agreement": score, "joint_coverage": coverage})
            results[key] = {"checkpoints": rows, "successive_stability": stability}
            fitted[key] = models[-1]

    final_residual = residuals[-1]
    one_step_rmse = float(np.mean(data["one_step_rmse"]))
    tail_steps = data["target_indices"][:, -1] - data["target_indices"][:, 0]
    for key, model in fitted.items():
        final = results[key]["checkpoints"][-1]
        final["cluster_composition"] = []
        if model is None:
            final["minimum_between_basin_rms"] = None
            final["one_step_to_basin_ratio"] = None
            continue
        for c in range(model.n_clusters):
            member = model.labels == c
            final["cluster_composition"].append({
                "cluster": c,
                "intact": int(((truth == 0) & member).sum()),
                "toppled": int(((truth == 1) & member).sum()),
            })
        noise = model.labels < 0
        final["noise_composition"] = {
            "intact": int(((truth == 0) & noise).sum()),
            "toppled": int(((truth == 1) & noise).sum()),
        }
        if model.n_clusters >= 2:
            centers_raw = np.stack([
                final_residual[model.labels == c].mean(0) for c in range(model.n_clusters)])
            pair = np.sqrt(((centers_raw[:, None] - centers_raw[None]) ** 2).mean(-1))
            between_rms = float(pair[np.triu_indices(len(pair), 1)].min())
        else:
            between_rms = float("nan")
        final["minimum_between_basin_rms"] = between_rms
        final["one_step_to_basin_ratio"] = float(one_step_rmse / between_rms)

    primary_key = f"pca{pca_dims[0]}_min{min_fractions[0]:g}"
    primary_final = results[primary_key]["checkpoints"][-1]
    return {
        "truth_counts": {"intact": int((truth == 0).sum()), "toppled": int(truth.sum())},
        "peak_tilt_deg": {"min": float(tilts.min()), "max": float(tilts.max())},
        "rollout_finite": bool(np.isfinite(data["endings"]).all()),
        "nominal_tail_steps": {
            "min": int(tail_steps.min()),
            "median": float(np.median(tail_steps)),
            "max": int(tail_steps.max()),
        },
        "one_step_rmse": one_step_rmse,
        "one_step_normalised_rmse": float(np.mean(data["one_step_nrmse"])),
        "minimum_between_basin_rms": primary_final["minimum_between_basin_rms"],
        "one_step_to_basin_ratio": primary_final["one_step_to_basin_ratio"],
        "primary": primary_key,
        "configurations": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--labels", default=str(ROOT / "data" / "jenga" / "labels_noise100.json"))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--tail-fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    ap.add_argument("--pca", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--min-cluster-fractions", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default=str(
        ROOT / "results" / "jenga" / "j2_j3_predicted_endings.npz"))
    ap.add_argument("--output", default=str(
        ROOT / "results" / "jenga" / "j2_j3_predicted_basins.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    fractions = args.tail_fractions
    if sorted(set(fractions)) != fractions or fractions[0] != 0 or fractions[-1] != 1:
        ap.error("tail fractions must be unique, increasing, and span 0 to 1")

    cache = Path(args.cache)
    replay = JengaReplay(args.lmdb)
    ids = replay.episode_ids[:min(args.episodes, len(replay.episode_ids))]
    try:
        if args.reuse_cache and cache.exists():
            data = dict(np.load(cache, allow_pickle=False))
            if list(data["episode_ids"].astype(str)) != list(ids):
                raise ValueError("cache episode ids do not match this run")
            print(f"loaded {cache}")
        else:
            model = load_world_model(args.checkpoint, args.device)
            print(f"rolling {len(ids)} episodes with nominal continuation on {args.device}")
            data = collect(args, replay, ids, model)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **data)
            print(f"cached predicted endings -> {cache}")
    finally:
        replay.close()

    labels = json.load(open(args.labels))
    result = analyse(data, labels, args.pca, args.min_cluster_fractions)
    result.update({
        "checkpoint": portable(args.checkpoint), "lmdb": portable(args.lmdb),
        "labels": portable(args.labels), "episodes": len(ids), "horizon": args.horizon,
        "tail_fractions": fractions, "pca_dims": args.pca,
        "min_cluster_fractions": args.min_cluster_fractions,
    })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("\nJ2/J3 summary")
    print(f"one-step RMSE {result['one_step_rmse']:.4f}; minimum between-basin RMS "
          f"{result['minimum_between_basin_rms']:.4f}; ratio "
          f"{result['one_step_to_basin_ratio']:.3f}")
    for key, cfg in result["configurations"].items():
        final = cfg["checkpoints"][-1]
        late = cfg["successive_stability"][-1]
        sep = "—" if final["separation_ratio"] is None else f"{final['separation_ratio']:.2f}"
        error_ratio = ("—" if final["one_step_to_basin_ratio"] is None
                       else f"{final['one_step_to_basin_ratio']:.2f}")
        print(f"{key:>16}: final k={final['n_clusters']}, agreement={final['agreement']:.1%}, "
              f"coverage={final['coverage']:.1%}, sep={sep}, error/basin={error_ratio}, "
              f"90%-to-full stability={late['agreement']:.1%} "
              f"on {late['joint_coverage']:.1%}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
