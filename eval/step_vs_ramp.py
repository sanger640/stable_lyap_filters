"""Step-vs-ramp diagnostic (descriptive; held-out TRAIN D0 states only).

Hypothesis: real physics maps the sampled uncertainty onto outcomes with a STEP (a threshold: stays or
topples, nothing between), while a smooth learned dynamics model maps it with a RAMP (intermediate
"half-toppled" outcomes), which fills the gap between the two outcome groups.

For each D0 fork state (2-62 of 64 real futures topple a neighbour):
  push   = out-of-fold logistic-regression score of the 42 sampled error numbers (perception draw 18 +
           execution snippet 24) predicting the REAL topple label (fitted on real outcomes only);
  y      = ending at hold 30 projected on the real futures' first principal axis, scaled so the real
           'stays' group sits at -1 and the 'topples' group at +1 (same transform for every model);
  fit    y ~ tanh((push - s0) / tau): tau is the transition width (small = step), s0 the threshold.
Models get exactly the real futures' inputs: the rendered perturbed start latent of each future and its
executed actions (iteration-1 deterministic model; iteration-2 deterministic model).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
DATA = ROOT / "results/program_b/step3b/d0"
RESULT = ROOT / "results/jenga/step_vs_ramp.json"


def push_scores(features, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folds = min(8, int(labels.sum()), int((~labels).sum()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    s = cross_val_predict(clf, features, labels, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                          method="decision_function")
    return (s - s.mean()) / (s.std() + 1e-9)


def fit_tanh(s, y):
    from scipy.optimize import curve_fit
    try:
        (s0, tau), _ = curve_fit(lambda x, s0, tau: np.tanh((x - s0) / tau), s, y, p0=[0.0, 1.0],
                                 bounds=([-5, 0.01], [5, 50]), maxfev=20000)
        return float(s0), float(tau)
    except RuntimeError:
        return float("nan"), float("nan")


def main():
    import torch
    import step3_world_model as wm
    from fresh_fragility_panel import topple_count
    import uncertainty_monitor_dev as um
    snippets = np.load(um.BENCH)["snippets"].reshape(64, -1)
    models = {"iter1_deterministic": wm.WorldModel.load(ROOT / "results/program_b/step3/wm_deterministic.pt"),
              "iter2_deterministic": wm.WorldModel.load(ROOT / "results/program_b/step3b/dyn_full.pt")}
    rows = []
    for path in sorted(DATA.glob("ep*.npz")):
        with np.load(path) as f:
            flags = np.array([topple_count(f["traces"][j:j + 1], f["start_tilt"]) > 0 for j in range(64)])
            if not 2 <= flags.sum() <= 62:
                continue
            features = np.concatenate([f["d_pos"].reshape(64, -1), f["d_rot"].reshape(64, -1), snippets], 1)
            push = push_scores(features, flags)
            real = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            mean = real.mean(0)
            axis = np.linalg.svd(real - mean, full_matrices=False)[2][0]
            pc = (real - mean) @ axis
            c0, c1 = pc[~flags].mean(), pc[flags].mean()
            norm = lambda p: (p - (c0 + c1) / 2) / ((c1 - c0) / 2)
            row = {"state": path.stem, "topples": int(flags.sum()),
                   "push_auc": float(((push[flags][:, None] > push[~flags][None]).mean()))}
            series = {"real": norm(pc)}
            starts = torch.as_tensor(f["z0k"], device=wm.DEVICE).float()
            for name, model in models.items():
                end = model.step  # deterministic: one call per model step
                z = starts.clone()
                pairs = torch.as_tensor(f["actions"], device=wm.DEVICE).reshape(64, wm.STEPS, -1)
                with torch.no_grad():
                    for t in range(wm.STEPS):
                        z = end(z, pairs[:, t])
                series[name] = norm(((z.reshape(64, -1).double().cpu().numpy()) - mean) @ axis)
            for name, y in series.items():
                s0, tau = fit_tanh(push, y)
                row[name] = {"tau": tau, "s0": s0, "mid_fraction": float(np.mean(np.abs(y) < 0.5)),
                             "spread": float(np.std(y)), "corr_with_real": float(np.corrcoef(y, series["real"])[0, 1])}
            row["series"] = {k: v.round(3).tolist() for k, v in series.items()}
            row["push"] = push.round(3).tolist()
        rows.append(row)
    names = ["real"] + list(models)
    summary = {"fork_states": len(rows),
               "median_push_auc_out_of_fold": float(np.median([r["push_auc"] for r in rows]))}
    for n in names:
        summary[n] = {k: float(np.nanmedian([r[n][k] for r in rows]))
                      for k in ("tau", "mid_fraction", "spread", "corr_with_real")}
        summary[n]["threshold_shift_vs_real"] = float(np.nanmedian([abs(r[n]["s0"] - r["real"]["s0"]) for r in rows]))
    RESULT.write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "summary": summary,
                                  "per_state": [{k: v for k, v in r.items() if k not in ("series", "push")}
                                                for r in rows]}, indent=1) + "\n")
    (ROOT / "results/program_b/step3b/step_vs_ramp_series.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
