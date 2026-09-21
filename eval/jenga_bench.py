"""The frozen Jenga fork benchmark: one cache, one manifest, one evaluator (PLAN_NEXT.md Phase 1).

    freeze   simulate once to each benchmark state and store everything evaluation needs --
             the exact start state, the 64 perturbed action windows per error scale, the real
             simulator endings of every probe, and the camera frames at the chunk start -- in one
             file, with a manifest of checksums. Refuses to overwrite a freeze without --force.
    eval     score any dynamics checkpoint against the frozen benchmark. Verifies the manifest
             AND that this file's evaluation constants still equal the frozen ones, and refuses to
             run on any mismatch, so no later edit can silently change the benchmark. Needs no
             simulator.

Development split: batch 1 (sets the pre-declared operating point). Test split: batch 3 (84 topple
forks at 1x). The score is the RMS spread of the three block positions (mm) over the 64 probes after
the 30-step hold. Fork labels are used only to grade; only quiet states set any threshold.

Output, per checkpoint: AUC; recall at 1/3/5/10% FPR (threshold swept on TEST quiet states -- a
selection diagnostic); recall and false positives at the pre-declared operating point (95th
percentile of DEVELOPMENT quiet scores) with episode-clustered intervals; a blind-fork indicator for
every fork; fork and quiet spread distributions; median contrast, predicted and real; predicted over
real separation; ending error against the simulator; the raw score of every state; the thresholds;
the seed; and a checkpoint / config hash.

With --probe the start state is ESTIMATED from the frozen frames (the V1 pipeline), with the gripper
taken from proprioception, so V1 runs on exactly the same benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

BENCH_DIR = ROOT / "results/jenga/bench"
BENCH_FILE = BENCH_DIR / "jenga_bench.npz"
MANIFEST_FILE = BENCH_DIR / "manifest.json"
SPLITS = {
    "dev": {"states": "results/jenga/holdout_stage2_shared.json",
            "stage0": "results/jenga/holdout_stage0.json",
            "cache": "results/jenga/holdout_stage0_cache.npz", "reset_base": None},
    "test": {"states": "results/jenga/holdout3_stage2_shared.json",
             "stage0": "results/jenga/holdout3_stage0.json",
             "cache": "results/jenga/holdout3_stage0_cache.npz", "reset_base": 1000},
}
SNIPPETS = "results/jenga/holdout_stage0_cache.npz"      # byte-identical to batch 3's snippets
SIM_ARCHIVE = "vendor/panda_express_sim.tar"
FRAMES = 3                                    # NUM_HIST frames ending at the chunk start

# Every constant that decides a number. Frozen into the manifest; `eval` refuses to run if these no
# longer match, so changing one requires a deliberate re-freeze.
EVAL_CONFIG = {
    "scales": [0.5, 1.0, 2.0],
    "probes": 64,
    "hold_step": 30,
    "score": "RMS spread of the 3 block positions (mm) over the probes after the hold",
    "positive_class": "topple_fork",
    "negative_class": "quiet",
    "operating_point": "per scale, 95th percentile (numpy method='higher') of DEV quiet scores; "
                       "alarm = score > threshold",
    "budget": 0.05,
    "matched_fpr": [0.01, 0.03, 0.05, 0.10],
    "matched_threshold": "quantile of TEST quiet scores at 1 - fpr, method='higher'",
    "blind_margin": 0.5,
    "blind": "a fork whose score is below blind_margin x the pre-declared threshold",
    "own_hold": False,
    "bootstrap_resamples_rate": 2000,
    "bootstrap_resamples_auc": 1000,
    "bootstrap_seed": 0,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
}


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def spread(endings):
    """(probes, 9) mm -> RMS distance of the probes from their mean."""
    return float(np.sqrt(np.mean(np.sum((endings - endings.mean(0)) ** 2, 1))))


# ----------------------------------------------------------------------------------------- freeze

def freeze_split(name, spec, snippets, xml):
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim
    from jenga_stage0_noise_oracle import RECORD_AT
    from jenga_stage3_predicted_forks import action_windows
    from jenga_state_data import step_state

    rows = json.loads((ROOT / spec["states"]).read_text())["rows"]
    stage0 = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
    index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0)}
    pose = np.load(ROOT / spec["cache"], allow_pickle=False)["pose"]
    hold_at = RECORD_AT.index(EVAL_CONFIG["hold_step"])
    scales = EVAL_CONFIG["scales"]
    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r

    out = {k: [] for k in ("episode", "chunk", "classes", "topples", "start", "windows",
                           "real_mm", "frames")}
    replay = JengaReplay(DEFAULT_LMDB)
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id)
            base = spec["reset_base"]
            sim.reset(int(episode_id) + base if base is not None else int(episode_id))
            history = [sim.render()]
            for step, action in enumerate(episode.actions):
                if step in wanted[episode_id]:
                    row = wanted[episode_id][step]
                    frames = history[-FRAMES:]
                    while len(frames) < FRAMES:
                        frames = [frames[0]] + frames
                    cache_row = index[(episode_id, step)]
                    out["episode"].append(str(episode_id))
                    out["chunk"].append(int(step))
                    out["classes"].append([row["by_scale"][str(s)]["class"] for s in scales])
                    out["topples"].append([int(row["by_scale"][str(s)]["topple_count"])
                                           for s in scales])
                    out["start"].append(step_state(sim))
                    out["windows"].append(np.stack([
                        action_windows(episode.actions, step, snippets, s, own_hold=False)
                        for s in scales]))
                    out["real_mm"].append(np.stack([
                        1000 * pose[cache_row, k, :, hold_at, 0:9] for k in range(len(scales))]))
                    out["frames"].append(np.stack(frames))
                sim.execute(action)
                history.append(sim.render())
                history = history[-FRAMES:]
            print(f"  freeze {name}: episode {number + 1}/{len(wanted)}", flush=True)
    finally:
        sim.close()
        replay.close()
    return {f"{name}_episode": np.asarray(out["episode"]),
            f"{name}_chunk": np.asarray(out["chunk"], np.int64),
            f"{name}_classes": np.asarray(out["classes"]),
            f"{name}_topples": np.asarray(out["topples"], np.int64),
            f"{name}_start": np.stack(out["start"]).astype(np.float32),
            f"{name}_windows": np.stack(out["windows"]).astype(np.float32),
            f"{name}_real_mm": np.stack(out["real_mm"]).astype(np.float32),
            f"{name}_frames": np.stack(out["frames"]).astype(np.uint8)}


def environment():
    import mujoco
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "mujoco": mujoco.__version__,
            "platform": platform.platform()}


def freeze(args):
    from jenga_short_held_tails import extract_sim
    if BENCH_FILE.exists() and not args.force:
        raise SystemExit(f"{BENCH_FILE} exists; the benchmark is frozen. Re-freeze only "
                         "deliberately, with --force.")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    snippets = np.load(ROOT / SNIPPETS, allow_pickle=False)["snippets"]
    arrays = {"snippets": snippets}
    with tempfile.TemporaryDirectory(prefix="jenga_bench_") as temp:
        xml = str(extract_sim(ROOT / SIM_ARCHIVE, temp))
        for name, spec in SPLITS.items():
            arrays.update(freeze_split(name, spec, snippets, xml))
    np.savez_compressed(BENCH_FILE, **arrays)

    classes, starts = arrays["test_classes"], arrays["test_start"]
    one = EVAL_CONFIG["scales"].index(1.0)
    fork_rows = classes[:, one] == EVAL_CONFIG["positive_class"]
    quiet_rows = classes[:, one] == EVAL_CONFIG["negative_class"]
    counts = {name: {str(s): {c: int(n) for c, n in zip(*np.unique(
        arrays[f"{name}_classes"][:, k], return_counts=True))}
        for k, s in enumerate(EVAL_CONFIG["scales"])} for name in SPLITS}
    manifest = {
        "benchmark": "Jenga execution-noise fork benchmark (PLAN_NEXT.md Phase 1)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bench_file": str(BENCH_FILE.relative_to(ROOT)),
        "bench_sha256": sha256_file(BENCH_FILE),
        "eval_config": EVAL_CONFIG,
        "components_sha256": {
            "test_fork_states_1x": sha256_array(starts[fork_rows]),
            "test_quiet_states_1x": sha256_array(starts[quiet_rows]),
            "dev_states": sha256_array(arrays["dev_start"]),
            "perturbation_set_snippets": sha256_array(snippets),
            "perturbation_windows_test": sha256_array(arrays["test_windows"]),
            "perturbation_windows_dev": sha256_array(arrays["dev_windows"]),
            "oracle_endings_test": sha256_array(arrays["test_real_mm"]),
            "oracle_endings_dev": sha256_array(arrays["dev_real_mm"]),
            "frames_test": sha256_array(arrays["test_frames"]),
            "frames_dev": sha256_array(arrays["dev_frames"]),
        },
        "sources_sha256": {path: sha256_file(ROOT / path) for path in sorted(
            {SNIPPETS, SIM_ARCHIVE} | {v for s in SPLITS.values() for k, v in s.items()
                                       if k in ("states", "stage0", "cache")})},
        "splits": {name: {"states": int(len(arrays[f"{name}_start"])),
                          "episodes": int(len(set(arrays[f"{name}_episode"]))),
                          "reset_seed_base": spec["reset_base"],
                          "classes_by_scale": counts[name]} for name, spec in SPLITS.items()},
        "environment": environment(),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"frozen: {BENCH_FILE} ({BENCH_FILE.stat().st_size / 1e6:.0f} MB), "
          f"manifest {MANIFEST_FILE}")


# ------------------------------------------------------------------------------------------- eval

def verify():
    """Refuse to evaluate unless the frozen file and this file's constants are unchanged."""
    if not MANIFEST_FILE.exists() or not BENCH_FILE.exists():
        raise SystemExit("no frozen benchmark: run `freeze` first")
    manifest = json.loads(MANIFEST_FILE.read_text())
    actual = sha256_file(BENCH_FILE)
    if actual != manifest["bench_sha256"]:
        raise SystemExit(f"benchmark file changed since it was frozen:\n  frozen "
                         f"{manifest['bench_sha256']}\n  now    {actual}")
    if json.loads(json.dumps(EVAL_CONFIG)) != manifest["eval_config"]:
        changed = sorted(k for k in set(EVAL_CONFIG) | set(manifest["eval_config"])
                         if json.loads(json.dumps(EVAL_CONFIG)).get(k)
                         != manifest["eval_config"].get(k))
        raise SystemExit(f"evaluation constants differ from the frozen ones: {changed}")
    return manifest


def checkpoint_identity(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = {k: v for k, v in state.items()
              if isinstance(v, (int, float, str, bool, type(None)))}
    return {"path": str(Path(path).resolve().relative_to(ROOT)) if str(Path(path).resolve())
            .startswith(str(ROOT)) else str(path),
            "sha256": sha256_file(path), "config": config,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode())
            .hexdigest(), "seed": config.get("seed")}


def score_split(name, bench, model, scale, device, estimator=None, encoder=None, pixels=196):
    from jenga_v1_gate import encode_frames
    from jenga_v1_probe import GROUPS
    from jenga_w5_eval import hold_index, predict
    starts = bench[f"{name}_start"]
    windows = bench[f"{name}_windows"]
    real = bench[f"{name}_real_mm"]
    rows = []
    for i in range(len(starts)):
        start = starts[i]
        position_error = None
        if estimator is not None:
            frames = list(bench[f"{name}_frames"][i][-estimator.frames:])
            estimated = estimator(encode_frames(encoder, device, frames, pixels))
            estimated[GROUPS["gripper"][0]] = start[GROUPS["gripper"][0]]   # proprioception
            position_error = float(np.sqrt(np.mean((1000 * (estimated[:9] - start[:9])) ** 2)))
            start = estimated
        for k, s in enumerate(EVAL_CONFIG["scales"]):
            ends = 1000 * predict(model, scale, start, windows[i, k], device)[
                :, hold_index(model, EVAL_CONFIG["hold_step"]), 0:9]
            truth = real[i, k]
            rows.append({"split": name, "episode_id": str(bench[f"{name}_episode"][i]),
                         "chunk_start": int(bench[f"{name}_chunk"][i]), "scale": str(s),
                         "class": str(bench[f"{name}_classes"][i, k]),
                         "topple_count": int(bench[f"{name}_topples"][i, k]),
                         "score": spread(ends), "real_spread_mm": spread(truth),
                         "ending_error_mm": float(np.sqrt(np.mean(np.sum(
                             (ends - truth) ** 2, 1)))),
                         "state_position_error_mm": position_error})
    return rows


def summarise(dev, test):
    from jenga_w5_gate3 import clustered_rate
    from jenga_w6_curves import auc, clustered_auc, recall_at_fpr
    cfg = EVAL_CONFIG
    pos, neg = cfg["positive_class"], cfg["negative_class"]
    out = {}
    for s in map(str, cfg["scales"]):
        dev_quiet = np.array([r["score"] for r in dev if r["scale"] == s and r["class"] == neg])
        threshold = float(np.quantile(dev_quiet, 1 - cfg["budget"], method="higher"))
        part = [r for r in test if r["scale"] == s]
        for r in part:
            r["threshold"] = threshold
            r["alarm"] = bool(r["score"] > threshold)
            r["blind"] = bool(r["class"] == pos and r["score"] < cfg["blind_margin"] * threshold)
        forks = [r for r in part if r["class"] == pos]
        quiet = [r for r in part if r["class"] == neg]
        f = np.array([r["score"] for r in forks]); q = np.array([r["score"] for r in quiet])
        fr = np.array([r["real_spread_mm"] for r in forks])
        qr = np.array([r["real_spread_mm"] for r in quiet])
        quant = {f"p{int(round(100 * v))}": None for v in cfg["quantiles"]}
        entry = {
            "fork_states": len(forks), "quiet_states": len(quiet),
            "pre_declared": {
                "threshold_mm": threshold, "dev_quiet_states": int(len(dev_quiet)),
                "dev_false_positive": float(np.mean(dev_quiet > threshold)),
                "recall": clustered_rate(forks, "alarm", n_boot=cfg["bootstrap_resamples_rate"],
                                         seed=cfg["bootstrap_seed"]),
                "false_positive": clustered_rate(quiet, "alarm",
                                                 n_boot=cfg["bootstrap_resamples_rate"],
                                                 seed=cfg["bootstrap_seed"])},
            "auc": auc(f, q) if len(f) and len(q) else None,
            "auc_ci95": clustered_auc([(r["episode_id"], r["score"], r["class"] == pos)
                                       for r in forks + quiet],
                                      n_boot=cfg["bootstrap_resamples_auc"],
                                      seed=cfg["bootstrap_seed"]) if len(f) and len(q) else None,
            "matched_fpr": {},
            "blind_forks": {"count": int(sum(r["blind"] for r in forks)),
                            "rate": float(np.mean([r["blind"] for r in forks])) if forks else None},
            "fork_spread_mm": {k: float(np.quantile(f, v)) for k, v in
                               zip(quant, cfg["quantiles"])} if len(f) else None,
            "quiet_spread_mm": {k: float(np.quantile(q, v)) for k, v in
                                zip(quant, cfg["quantiles"])} if len(q) else None,
            "contrast_predicted": float(np.median(f) / max(np.median(q), 1e-12))
            if len(f) and len(q) else None,
            "contrast_real": float(np.median(fr) / max(np.median(qr), 1e-12))
            if len(f) and len(q) else None,
            "separation_ratio_fork": float(f.sum() / max(fr.sum(), 1e-12)) if len(f) else None,
            "separation_ratio_quiet": float(q.sum() / max(qr.sum(), 1e-12)) if len(q) else None,
            "ending_error_mm_median": {
                name: float(np.median([r["ending_error_mm"] for r in group])) if group else None
                for name, group in (("fork", forks), ("quiet", quiet))},
        }
        for target in cfg["matched_fpr"]:
            value, t = recall_at_fpr(f, q, target)
            key = f"{int(round(100 * target))}pct"
            entry["matched_fpr"][key] = {"recall": value, "threshold_mm": t}
        errors = [r["state_position_error_mm"] for r in part
                  if r["state_position_error_mm"] is not None]
        if errors:
            entry["state_position_error_mm_median"] = float(np.median(errors))
        out[s] = entry
    return out


def evaluate(args):
    from jenga_w5_eval import load_model
    manifest = verify()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = dict(np.load(BENCH_FILE, allow_pickle=False))
    model, scale = load_model(args.model, device)
    estimator = encoder = None
    if args.probe:
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        from jenga_v1_gate import Estimator
        estimator = Estimator(args.probe, args.probe_kind, device)
        encoder = load_world_model(DEFAULT_CHECKPOINT, device).encoder
    dev = score_split("dev", bench, model, scale, device, estimator, encoder, args.pixels)
    test = score_split("test", bench, model, scale, device, estimator, encoder, args.pixels)
    summary = summarise(dev, test)

    identity = checkpoint_identity(args.model)
    training = Path(args.model).with_name(Path(args.model).stem + "_train.json")
    rollout_error = None
    if training.exists():
        report = json.loads(training.read_text())
        history = report.get("rollout_history") or report.get("history") or []
        if history and "validation" in history[-1]:
            rollout_error = history[-1]["validation"].get("rollout_state_error")
    result = {
        "benchmark_sha256": manifest["bench_sha256"],
        "eval_config": EVAL_CONFIG,
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": identity,
        "pipeline": "V1: DINO -> probe -> state (gripper from proprioception) -> dynamics"
                    if args.probe else "privileged start state -> dynamics",
        "probe": {"path": args.probe, "sha256": sha256_file(args.probe), "kind": args.probe_kind,
                  "pixels": args.pixels} if args.probe else None,
        "validation_rollout_state_error": rollout_error,
        "scales": summary,
        "rows": test + dev,
    }
    output = Path(args.output) if args.output else (
        ROOT / "results/jenga/bench_eval" / f"{Path(args.model).stem}"
        f"{'_v1' if args.probe else ''}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    one = summary["1.0"]
    matched = " / ".join(f"{100 * v['recall']:.0f}" for v in one["matched_fpr"].values())
    print(f"{identity['path']}  seed {identity['seed']}  -> {output}")
    print(f"  1x: pre-declared recall {100 * one['pre_declared']['recall']['rate']:.0f}% "
          f"FP {100 * one['pre_declared']['false_positive']['rate']:.0f}%  AUC {one['auc']:.3f}  "
          f"matched 1/3/5/10% {matched}  "
          f"blind {one['blind_forks']['count']}/{one['fork_states']}  "
          f"quiet p99 {one['quiet_spread_mm']['p99']:.2f} mm  contrast "
          f"{one['contrast_predicted']:.1f}")
    two = summary["2.0"]["pre_declared"]
    print(f"  2x: pre-declared recall {100 * two['recall']['rate']:.0f}% "
          f"FP {100 * two['false_positive']['rate']:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="build the frozen cache and manifest (once)")
    f.add_argument("--force", action="store_true")
    e = sub.add_parser("eval", help="score one checkpoint against the frozen benchmark")
    e.add_argument("--model", required=True)
    e.add_argument("--probe", default=None, help="V1 state probe; omit for privileged state")
    e.add_argument("--probe-kind", choices=("linear", "mlp"), default="linear")
    e.add_argument("--pixels", type=int, default=196)
    e.add_argument("--output", default=None)
    sub.add_parser("verify", help="check the frozen file and constants, then exit")
    args = ap.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        manifest = verify()
        print(f"benchmark intact: {manifest['bench_sha256'][:16]}... frozen {manifest['created']}")


if __name__ == "__main__":
    main()
