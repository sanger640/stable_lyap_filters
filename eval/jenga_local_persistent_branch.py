"""Held-out evaluation of an unlabeled local branch margin on physical probes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from local_persistent_branch import detect_local_persistent_branch  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402


def fit_unlabeled_projection(original_cache, excluded_episodes, dimensions=6):
    data = np.load(original_cache, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    keep = np.asarray([r["episode_id"] not in excluded_episodes for r in metadata])
    source = original_physical_responses(original_cache)[keep]
    scale = robust_component_scale(source)
    x = (source / scale).reshape(-1, source.shape[-1])
    mean = x.mean(axis=0)
    components = np.linalg.svd(x - mean, full_matrices=False)[2][:dimensions]
    return scale, mean, components, int(keep.sum())


def summarise(rows):
    summary = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == stratum]
        oracle_found = [r for r in part if r["oracle_margin"]["status"] == "bracketed"]
        matched = [r for r in oracle_found if r["local_margin"]["status"] == "bracketed"]
        summary[stratum] = {
            "states": len(part),
            "local_bracketed": int(sum(r["local_margin"]["status"] == "bracketed"
                                         for r in part)),
            "oracle_bracketed": len(oracle_found),
            "oracle_edge_unconfirmed": int(sum(
                r["oracle_margin"]["status"] == "edge_unconfirmed" for r in part)),
            "oracle_bracketed_with_local_bracket": len(matched),
            "median_abs_upper_mm_error_on_matched": (float(np.median([
                abs(r["local_margin"]["upper_mm"] - r["oracle_margin"]["upper_mm"])
                for r in matched])) if matched else None),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/persistent_branch_margin_cache.npz"))
    ap.add_argument("--oracle-result", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/local_persistent_branch.json"))
    args = ap.parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    metadata = json.loads(str(cache["metadata_json"]))
    coefficients = cache["coefficients"]
    endpoints = cache["endpoints"]
    spans = cache["span_mm"]
    oracle = json.loads(Path(args.oracle_result).read_text())
    oracle_rows = {(r["episode_id"], r["chunk_start"]): r for r in oracle["rows"]}
    excluded = {r["episode_id"] for r in metadata}
    scale, mean, components, reference_count = fit_unlabeled_projection(
        args.original_cache, excluded)
    rows = []
    for meta, response, span in zip(metadata, endpoints, spans):
        projected = ((response / scale - mean) @ components.T)
        margin = asdict(detect_local_persistent_branch(coefficients, projected))
        if margin["status"] == "bracketed":
            margin["lower_mm"] = margin["lower_abs_coefficient"] * float(span)
            margin["upper_mm"] = margin["upper_abs_coefficient"] * float(span)
        key = (meta["episode_id"], meta["chunk_start"])
        row = oracle_rows[key]
        rows.append({**meta, "chunk_span_mm": float(span), "local_margin": margin,
                     "oracle_margin": row["oracle_margin"],
                     "prior_trajectory_alarm": row["prior_trajectory_alarm"]})
    result = {"protocol": {"labels_used_to_fit_or_score": False,
                            "oracle_used_only_after_scoring": True,
                            "reference_original_states": reference_count,
                            "test_episodes_excluded_from_projection": True,
                            "projection": "unlabeled robust component scaling + PCA6",
                            "local_model": "continuous piecewise linear vs same plus step; "
                                           "BIC per gap using up to 4 probes per side",
                            "persistence": "same split gap has positive step evidence at both "
                                           "held tails 5 and 10",
                            "strength_grid": coefficients.tolist(),
                            "no_branch_note": "not resolved on finite grid is not proof of safety"},
              "summary": summarise(rows), "rows": rows}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
