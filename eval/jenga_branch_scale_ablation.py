"""Post-hoc feature ablation of the frozen refinement intervals (not a new holdout)."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from branch_scale import compare_scales  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402


FEATURES = {"full": slice(None), "pose": slice(0, 18),
            "position": slice(0, 6), "rotation": slice(6, 18),
            "contacts": slice(18, 27)}


def projection(original_cache, excluded_episodes, feature_slice):
    source = np.load(original_cache, allow_pickle=False)
    metadata = json.loads(str(source["metadata_json"]))
    keep = np.asarray([r["episode_id"] not in excluded_episodes for r in metadata])
    reference = original_physical_responses(original_cache)[keep, :, feature_slice]
    scale = robust_component_scale(reference)
    flat = (reference / scale).reshape(-1, reference.shape[-1])
    mean = flat.mean(0)
    components = np.linalg.svd(flat - mean, full_matrices=False)[2][:6]
    return lambda values: (values[..., feature_slice] / scale - mean) @ components.T


def main():
    output = ROOT / "results/jenga/branch_scale_ablation.json"
    result = json.loads((ROOT / "results/jenga/branch_scale_refine.json").read_text())
    cache = np.load(ROOT / "results/jenga/branch_scale_refine_cache.npz", allow_pickle=False)
    grid = np.load(ROOT / "results/jenga/persistent_branch_margin_cache.npz")["coefficients"]
    original = ROOT / "results/jenga/local_delta_probe_arrays.npz"
    excluded = {r["episode_id"] for r in json.loads(
        (ROOT / "results/jenga/local_persistent_branch.json").read_text())["rows"]}
    # Reconstruct each chosen interval from the cached starting and midpoint endpoints.
    by_feature = {}
    for name, feature_slice in FEATURES.items():
        transform = projection(original, excluded, feature_slice)
        rows = []
        for i, row in enumerate(result["rows"]):
            start = cache["replayed_endpoints"][i]
            extra = cache["extra_responses"][i]
            intervals = row["intervals"]
            a, b = start
            split = row["candidate_margin"]["split_index"]
            original_left = float(grid[split - 1])
            original_right = float(grid[split])
            pairs = [(a, b)]
            current_left, current_right = a, b
            current_a, current_b = original_left, original_right
            for midpoint, interval in zip(extra, intervals):
                if np.isclose(interval[0], current_a):
                    current_b, current_right = (current_a + current_b) / 2, midpoint
                else:
                    current_a, current_left = (current_a + current_b) / 2, midpoint
                pairs.append((current_left, current_right))
            distances = np.stack([
                np.linalg.norm(transform(right) - transform(left), axis=1)
                for left, right in pairs], axis=1)
            decision = compare_scales(distances)
            rows.append({"stratum": row["stratum"],
                         "status": decision.status,
                         "retained_fraction_by_tail": decision.retained_fraction_by_tail})
        by_feature[name] = {stratum: {"candidates": int(sum(r["stratum"] == stratum for r in rows)),
                                     "persistent_separation": int(sum(
                                         r["stratum"] == stratum
                                         and r["status"] == "persistent_separation" for r in rows))}
                            for stratum in sorted({r["stratum"] for r in rows})}
    output.write_text(json.dumps({"warning": "post-hoc ablation on same selected intervals, "
                                              "not a fresh detector or holdout",
                                  "summary": by_feature}, indent=2) + "\n")
    print(json.dumps(by_feature, indent=2))


if __name__ == "__main__":
    main()
