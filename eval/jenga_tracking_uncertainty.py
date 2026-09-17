"""Fit a label-free tracking-error proxy without test episodes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from action_uncertainty import fit_tracking_error, tracking_arrays  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--test-panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/tracking_uncertainty.json"))
    args = ap.parse_args()
    panel = json.loads(Path(args.test_panel).read_text())
    excluded = {r["episode_id"] for r in panel["rows"]}
    replay = JengaReplay(args.lmdb)
    try:
        training = [replay.episode(ep) for ep in replay.episode_ids if ep not in excluded]
        testing = [replay.episode(ep) for ep in replay.episode_ids if ep in excluded]
        model = fit_tracking_error(training)
        design, errors = tracking_arrays(testing)
    finally:
        replay.close()
    residual = errors - design @ model.coefficients
    radii = np.sqrt(np.einsum("ni,ij,nj->n", residual,
                              np.linalg.inv(model.covariance), residual))
    axes = model.axes
    result = {"protocol": {"alignment": "command at t vs observed proprio at t+1",
                            "lag_model": "intercept plus target displacement since prior step",
                            "test_episodes_excluded_from_fit": True,
                            "proxy_warning": "includes model mismatch and unobserved controller effects; "
                                             "not a certified execution-error bound",
                            "coverage_quantile": .90},
              "training_episodes": len(training), "testing_episodes": len(testing),
              "training_samples": model.samples, "testing_samples": len(residual),
              "lag_coefficients": model.coefficients.tolist(),
              "residual_covariance_m2": model.covariance.tolist(),
              "one_sigma_axes_m": axes.tolist(),
              "training_radial_quantile_90": model.radial_quantile_90,
              "heldout_radial_coverage_at_training_q90": float(np.mean(
                  radii <= model.radial_quantile_90)),
              "heldout_residual_median_mm": float(1000 * np.median(
                  np.linalg.norm(residual, axis=1))),
              "heldout_residual_p90_mm": float(1000 * np.quantile(
                  np.linalg.norm(residual, axis=1), .90))}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("lag_coefficients", "residual_covariance_m2",
                                   "one_sigma_axes_m")}, indent=2))


if __name__ == "__main__":
    main()
