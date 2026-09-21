"""Throughput of the Step-7 trainer vs number of concurrent processes (speed audit).

For each configuration and each concurrency k, launches k identical-recipe training processes
together (different seeds, so they are real independent jobs), each for --batches batches of one
epoch, and reads the batch-loop time each one reports (`batch_seconds`: excludes data loading and
validation). Converts that to seeds/hour for a full run:

    seconds/seed ~= epochs x batches_per_epoch x (batch_seconds / batches) + per-epoch validation
    seeds/hour   = k x 3600 / seconds/seed

Only options that are bit-identical to the current code are benchmarked (see NOTES.md, speed audit).
Run on an otherwise idle GPU.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {"d1_current": ["--cw-loss", "branch"],
           "d1_batched": ["--cw-loss", "branch", "--fast-branch"],
           "d2_current": ["--cw-loss", "intervention"]}


def run_group(config, k, batches, workdir):
    procs = []
    for i in range(k):
        out = Path(workdir) / f"{config}_k{k}_{i}"
        cmd = [sys.executable, str(ROOT / "eval/jenga_w6_simple.py"), "--seed", str(100 + i),
               "--noise", "0.5", "--epochs", "1", "--max-batches", str(batches),
               "--cw-data", str(ROOT / "results/jenga/cw_data"), *CONFIGS[config],
               "--output", f"{out}.pt", "--report", f"{out}.json"]
        procs.append((out, subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)))
    for _, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"{config} k={k}: a benchmark process failed")
    history = [json.loads(Path(f"{o}.json").read_text())["history"][0] for o, _ in procs]
    return {"batch_seconds": [h["batch_seconds"] for h in history],
            "validation_seconds": [h["seconds"] - h["batch_seconds"] for h in history]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--batches-per-epoch", type=int, default=1573,
                    help="full-epoch batch count of the Step-7 D1/D2 runs (logged by the trainer)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default=str(ROOT / "results/jenga/step7_speed.json"))
    args = ap.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="jenga_speed_") as workdir:
        for config in CONFIGS:
            for k in args.concurrency:
                t0 = time.time()
                r = run_group(config, k, args.batches, workdir)
                per_batch = max(r["batch_seconds"]) / args.batches        # slowest job governs
                validation = max(r["validation_seconds"])
                seed_seconds = args.epochs * (args.batches_per_epoch * per_batch + validation)
                entry = {"k": k, "seconds_per_batch": per_batch,
                         "validation_seconds_per_epoch": validation,
                         "hours_per_seed": seed_seconds / 3600,
                         "seeds_per_hour": k * 3600 / seed_seconds,
                         "wall_seconds": time.time() - t0, **r}
                results.setdefault(config, []).append(entry)
                print(f"{config:11s} k={k}: {per_batch:.3f} s/batch  "
                      f"{seed_seconds / 3600:5.2f} h/seed -> {entry['seeds_per_hour']:.2f} "
                      f"seeds/hour", flush=True)
    Path(args.output).write_text(json.dumps({"protocol": vars(args), "results": results},
                                            indent=1) + "\n")


if __name__ == "__main__":
    main()
