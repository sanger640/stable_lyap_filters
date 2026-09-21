"""Full score distributions and recall-vs-FPR, so the remaining failures can be attributed.

Median contrast says where the two distributions sit; recall is set by how much they OVERLAP.
This reads the per-state scores already written by `jenga_w6_report.py` (no re-simulation) and,
per arm, pools the seeds to report:

  distribution      quantiles of D_fork and D_quiet, in millimetres
  recall vs FPR     the whole curve, plus recall at MATCHED false-positive rates, which is the
                    fair way to compare arms whose Gate 3 operating points landed differently
  AUC               threshold-free separation, with an episode-clustered interval
  attribution       which of the three explanations the failures fit:
                      noisy_quiet_tail  a few quiet states score high and drag the threshold up
                                        (p99/p50 of D_quiet, and the recall recovered by dropping
                                        the worst 2% of quiet states)
                      weak_forks        forks the model scores below a typical quiet state
                      broad_overlap     the bulk of the two distributions genuinely interleaves

The threshold here is swept on TEST quiet states, so these numbers are a diagnostic and a model
SELECTION tool, not the pre-declared Gate 3 operating point (which is set on development data and
is what `jenga_w6_report.py` reports). Both are printed side by side. Only the quiet class is used
to set any threshold; fork labels are used to score, never to choose.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MATCHED_FPR = (0.01, 0.03, 0.05, 0.10)


def load_rows(paths, scale):
    """-> list of (episode_id, score_mm, is_fork) for the graded classes at one error size."""
    out = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        for r in data["rows"]:
            if r["scale"] != scale or r["class"] not in ("topple_fork", "quiet"):
                continue
            out.append((r["episode_id"], r["predicted_mm"], r["class"] == "topple_fork"))
    return out


def recall_at_fpr(fork, quiet, target):
    """Highest recall whose false-positive rate on quiet states is <= target."""
    if not len(quiet) or not len(fork):
        return None, None
    threshold = float(np.quantile(quiet, 1 - target, method="higher"))
    return float(np.mean(fork > threshold)), threshold


def auc(fork, quiet):
    if not len(fork) or not len(quiet):
        return None
    greater = (fork[:, None] > quiet[None, :]).sum() + 0.5 * (fork[:, None] == quiet[None, :]).sum()
    return float(greater / (len(fork) * len(quiet)))


def clustered_auc(rows, n_boot=1000, seed=0):
    """AUC with a 95% interval, resampling whole episodes."""
    episodes = sorted({e for e, _, _ in rows})
    by_episode = {e: [(s, f) for x, s, f in rows if x == e] for e in episodes}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = [v for e in rng.choice(episodes, len(episodes), replace=True) for v in by_episode[e]]
        fork = np.array([s for s, f in draw if f])
        quiet = np.array([s for s, f in draw if not f])
        value = auc(fork, quiet)
        if value is not None:
            boots.append(value)
    if not boots:
        return None
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def attribution(fork, quiet):
    """Which story fits: a noisy quiet tail, weak forks, or genuine broad overlap?"""
    quiet_median = float(np.median(quiet))
    base, _ = recall_at_fpr(fork, quiet, 0.05)
    # Drop the worst 2% of quiet states: if recall jumps, the threshold was hostage to a few.
    keep = quiet[quiet <= np.quantile(quiet, 0.98)]
    trimmed, _ = recall_at_fpr(fork, keep, 0.05)
    return {"quiet_p99_over_median": float(np.quantile(quiet, 0.99) / max(quiet_median, 1e-12)),
            "recall_at_5pct": base,
            "recall_if_worst_2pct_of_quiet_dropped": trimmed,
            "recovered_by_trimming": None if base is None else float(trimmed - base),
            "weak_forks_below_quiet_median": float(np.mean(fork < quiet_median)),
            "quiet_above_fork_median": float(np.mean(quiet > np.median(fork)))}


def summarise(rows):
    fork = np.array([s for _, s, f in rows if f])
    quiet = np.array([s for _, s, f in rows if not f])
    entry = {"fork_states": len(fork), "quiet_states": len(quiet),
             "fork_mm": {f"p{int(100 * q)}": float(np.quantile(fork, q)) for q in QUANTILES},
             "quiet_mm": {f"p{int(100 * q)}": float(np.quantile(quiet, q)) for q in QUANTILES},
             "median_contrast": float(np.median(fork) / max(np.median(quiet), 1e-12)),
             "auc": auc(fork, quiet), "auc_ci95": clustered_auc(rows),
             "recall_at_matched_fpr": {}, "attribution": attribution(fork, quiet)}
    for target in MATCHED_FPR:
        value, threshold = recall_at_fpr(fork, quiet, target)
        entry["recall_at_matched_fpr"][f"{int(100 * target)}pct"] = {
            "recall": value, "threshold_mm": threshold}
    # The full curve, thinned to the distinct quiet quantiles that define it.
    curve = []
    for target in np.linspace(0.005, 0.5, 40):
        value, threshold = recall_at_fpr(fork, quiet, float(target))
        curve.append({"fpr": float(target), "recall": value, "threshold_mm": threshold})
    entry["curve"] = curve
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True,
                    help="report.json files; grouped into arms by their name minus the seed")
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    arms = {}
    for path in args.reports:
        name = Path(path).name.replace("w6_", "").replace("_report.json", "")
        arms.setdefault(re.sub(r"_s\d+$", "", name), []).append(path)

    result = {"scale": args.scale, "protocol": {
        "scores": "predicted ending spread in mm over 64 probes, pooled across an arm's seeds",
        "threshold": "swept on TEST quiet states: a diagnostic and selection tool, not the "
                     "pre-declared Gate 3 operating point",
        "labels": "quiet class sets thresholds; fork labels only score"}, "arms": {}}
    for arm, paths in sorted(arms.items()):
        rows = load_rows(paths, args.scale)
        if not rows:
            continue
        result["arms"][arm] = dict(summarise(rows), seeds=len(paths))

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'arm':10s} {'seeds':>5s} {'AUC':>16s} | recall at matched FPR "
          f"{'1%':>6s} {'3%':>6s} {'5%':>6s} {'10%':>6s} | {'contrast':>8s}")
    for arm, e in result["arms"].items():
        m = e["recall_at_matched_fpr"]
        ci = e["auc_ci95"] or [float("nan")] * 2
        print(f"{arm:10s} {e['seeds']:5d} {e['auc']:.3f} [{ci[0]:.2f}-{ci[1]:.2f}] | "
              f"{'':21s}" + " ".join(f"{100 * m[k]['recall']:5.0f}%" for k in
                                     ("1pct", "3pct", "5pct", "10pct"))
              + f" | {e['median_contrast']:8.1f}")
    print("\nattribution (pooled):")
    for arm, e in result["arms"].items():
        a = e["attribution"]
        print(f"  {arm:10s} quiet p99/median {a['quiet_p99_over_median']:6.1f}  "
              f"recall +{100 * a['recovered_by_trimming']:4.1f} pts if worst 2% of quiet dropped  "
              f"weak forks {100 * a['weak_forks_below_quiet_median']:4.0f}%  "
              f"quiet above fork median {100 * a['quiet_above_fork_median']:4.0f}%")


if __name__ == "__main__":
    main()
