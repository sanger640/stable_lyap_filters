"""Revised step 7: is the blind spot a data-coverage problem or an objective problem?

Compares four arms, each scored by the frozen evaluator over its seeds:

  D0     one-step + 0.5 noise on the original data           (w6_gnn_n5_s*, existing)
  D0+CW  same loss, contact-window branches added as data    (w6_cw_d0_s*)
  D1+CW  + the existing branch-distance loss on those branches (w6_cw_d1_s*)
  D2+CW  + the intervention-consistency loss on those branches (w6_cw_d2_s*)

D0 -> D0+CW isolates coverage; D0+CW -> D1+CW the old branch loss; D0+CW -> D2+CW the new objective.

Per arm: robust-blind count recomputed across THAT arm's seeds (q >= 0.8 at the pre-declared point
and at a matched 5% test FPR, and under both); how many of D0's robust-blind forks it recovers;
recall at matched 1/3/5/10% FPR; AUC; quiet p99; fork spread p50 -- means and ranges over seeds.
Also, for reporting only (Track A), the miss rate on forks whose toppling neighbour starts upright,
the class `blind_fork_analysis.md` identified.
"""
import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

ARMS = {"D0": "w6_gnn_n5_s", "D0+CW": "w6_cw_d0_s", "D1+CW": "w6_cw_d1_s",
        "D2+CW": "w6_cw_d2_s"}
SCALE = "1.0"


def arm_runs(prefix):
    paths = [p for p in glob.glob(str(ROOT / f"results/jenga/bench_eval/{prefix}*.json"))
             if re.fullmatch(re.escape(prefix) + r"\d+\.json", Path(p).name)]
    return [json.loads(Path(p).read_text()) for p in sorted(paths)]


def upright_forks():
    """Forks whose toppling neighbour starts upright (< 8 deg), from the frozen start states."""
    from jenga_stage0_noise_oracle import RECORD_AT
    from state_dynamics import neighbour_tilt_deg
    bench = np.load(ROOT / "results/jenga/bench/jenga_bench.npz", allow_pickle=False)
    rows = json.loads((ROOT / "results/jenga/holdout3_stage0.json").read_text())["rows"]
    index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(rows)}
    cache = np.load(ROOT / "results/jenga/holdout3_stage0_cache.npz", allow_pickle=False)
    tilt, topple, hold = cache["tilt"], cache["topple"], RECORD_AT.index(30)
    out = set()
    for i, key in enumerate(zip(bench["test_episode"].tolist(), bench["test_chunk"].tolist())):
        if bench["test_classes"][i, 1] != "topple_fork":
            continue
        start = neighbour_tilt_deg(bench["test_start"][i][None])[0]
        toppled = topple[index[key], 1, :, hold]
        which = int(np.argmax((tilt[index[key], 1, toppled, hold, :] >= 45).sum(0)))
        if start[which] < 8.0:
            out.add(key)
    return out


def summarise(runs, upright):
    n = len(runs)
    misses, misses_matched = {}, {}
    for run in runs:
        matched = run["scales"][SCALE]["matched_fpr"]["5pct"]["threshold_mm"]
        for r in run["rows"]:
            if r["split"] != "test" or r["scale"] != SCALE or r["class"] != "topple_fork":
                continue
            key = (r["episode_id"], r["chunk_start"])
            misses[key] = misses.get(key, 0) + int(not r["alarm"])
            misses_matched[key] = misses_matched.get(key, 0) + int(not r["score"] > matched)
    robust = {k for k, v in misses.items() if v / n >= 0.8}
    robust_matched = {k for k, v in misses_matched.items() if v / n >= 0.8}

    def stat(fn):
        values = [fn(run["scales"][SCALE]) for run in runs]
        return {"mean": float(np.mean(values)), "min": float(np.min(values)),
                "max": float(np.max(values))}
    return {
        "seeds": sorted(run["model"]["seed"] for run in runs),
        "robust_blind": len(robust), "robust_blind_matched": len(robust_matched),
        "robust_blind_both": len(robust & robust_matched),
        "robust_blind_forks": sorted(f"{e}:{c}" for e, c in robust & robust_matched),
        "mean_miss_rate": float(np.mean(list(misses.values())) / n),
        "upright_miss_rate": float(np.mean([misses[k] for k in upright if k in misses]) / n),
        "other_miss_rate": float(np.mean([v for k, v in misses.items() if k not in upright]) / n),
        # the pre-declared (dev-set) threshold realises different test FPRs per arm; at matched 5%:
        "upright_miss_rate_matched": float(
            np.mean([misses_matched[k] for k in upright if k in misses_matched]) / n),
        "other_miss_rate_matched": float(
            np.mean([v for k, v in misses_matched.items() if k not in upright]) / n),
        "robust_blind_matched_upright": sum(1 for k in robust_matched if k in upright),
        "recall_matched": {p: stat(lambda s, p=p: s["matched_fpr"][p]["recall"])
                           for p in ("1pct", "3pct", "5pct", "10pct")},
        "pre_declared_recall": stat(lambda s: s["pre_declared"]["recall"]["rate"]),
        "pre_declared_fp": stat(lambda s: s["pre_declared"]["false_positive"]["rate"]),
        "auc": stat(lambda s: s["auc"]),
        "quiet_p99_mm": stat(lambda s: s["quiet_spread_mm"]["p99"]),
        "fork_p50_mm": stat(lambda s: s["fork_spread_mm"]["p50"]),
        "blind_forks_per_seed": stat(lambda s: s["blind_forks"]["count"]),
        "_misses": misses}


def paired(base_runs, runs):
    """Per-seed change from the same seed of the base arm (seeds present in both)."""
    base = {run["model"]["seed"]: run["scales"][SCALE] for run in base_runs}
    rows = []
    for run in sorted(runs, key=lambda r: r["model"]["seed"]):
        seed, s = run["model"]["seed"], run["scales"][SCALE]
        if seed not in base:
            continue
        b = base[seed]
        row = {"seed": seed}
        for label, fn in (("blind", lambda x: x["blind_forks"]["count"]),
                          ("auc", lambda x: x["auc"]),
                          ("quiet_p99_mm", lambda x: x["quiet_spread_mm"]["p99"]),
                          ("fork_p50_mm", lambda x: x["fork_spread_mm"]["p50"]),
                          *((f"recall_{p}", lambda x, p=p: x["matched_fpr"][p]["recall"])
                            for p in ("1pct", "3pct", "5pct", "10pct"))):
            row[label] = {"base": fn(b), "arm": fn(s), "change": fn(s) - fn(b)}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "results/jenga/step7_compare.json"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()
    upright = upright_forks()
    arms, runs_by_arm = {}, {}
    for name, prefix in ARMS.items():
        if name not in args.arms:
            continue
        runs = arm_runs(prefix)
        if runs:
            arms[name], runs_by_arm[name] = summarise(runs, upright), runs
    if "D0" in arms:
        baseline = {k for k, v in arms["D0"]["_misses"].items()
                    if v / len(arms["D0"]["seeds"]) >= 0.8}
        for summary in arms.values():
            n = len(summary["seeds"])
            summary["d0_robust_blind_still_robust"] = sum(
                1 for k in baseline if summary["_misses"].get(k, 0) / n >= 0.8)
            summary["d0_robust_blind_total"] = len(baseline)
    for summary in arms.values():
        summary.pop("_misses")
    for name in arms:
        if name != "D0" and "D0" in runs_by_arm:
            arms[name]["paired_vs_d0"] = paired(runs_by_arm["D0"], runs_by_arm[name])
    result = {"scale": SCALE, "upright_forks": len(upright), "arms": arms}
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")

    head = (f"{'arm':6s} {'seeds':>5s} | {'robust':>6s} {'both':>4s} {'D0 kept':>7s} | "
            f"{'@1%':>4s} {'@3%':>4s} {'@5%':>4s} {'@10%':>4s} | {'AUC':>5s} "
            f"{'qp99':>6s} {'fp50':>6s} | {'upright':>7s} {'other':>5s} | "
            f"{'@5% upright':>11s} {'other':>5s}")
    print(head)
    for name, s in arms.items():
        m = s["recall_matched"]
        print(f"{name:6s} {len(s['seeds']):5d} | {s['robust_blind']:6d} "
              f"{s['robust_blind_both']:4d} "
              f"{s.get('d0_robust_blind_still_robust', 0):3d}/"
              f"{s.get('d0_robust_blind_total', 0):<3d} | "
              + " ".join(f"{100 * m[p]['mean']:4.0f}" for p in ("1pct", "3pct", "5pct", "10pct"))
              + f" | {s['auc']['mean']:.3f} {s['quiet_p99_mm']['mean']:6.2f} "
              f"{s['fork_p50_mm']['mean']:6.2f} | {s['upright_miss_rate']:7.2f} "
              f"{s['other_miss_rate']:5.2f} | {s['upright_miss_rate_matched']:11.2f} "
              f"{s['other_miss_rate_matched']:5.2f}")
    for name, s in arms.items():
        if "paired_vs_d0" not in s:
            continue
        print(f"\n{name} vs D0, same seed (arm - D0): blind, AUC, quiet p99, fork p50, "
              f"recall @1/3/5/10%")
        for row in s["paired_vs_d0"]:
            c = {k: v["change"] for k, v in row.items() if k != "seed"}
            print(f"  s{row['seed']:<2d} blind {row['blind']['base']:3d}->{row['blind']['arm']:3d}"
                  f"  AUC {c['auc']:+.3f}  qp99 {c['quiet_p99_mm']:+7.2f}  "
                  f"fp50 {c['fork_p50_mm']:+7.2f}  recall "
                  + " ".join(f"{100 * c[f'recall_{p}']:+4.0f}"
                             for p in ("1pct", "3pct", "5pct", "10pct")))


if __name__ == "__main__":
    main()
