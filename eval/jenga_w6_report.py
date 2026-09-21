"""The four numbers for the three controlled models, inside realistic execution error only.

  ordinary rollout error   validation state error of the plain rollout (from the training report)
  fork recall              share of topple-fork states the monitor flags (Gate 3, batch 3)
  non-fork false positives share of quiet states it flags at the same operating point
  D_pred / D_real          predicted branch separation over real branch separation

The separation ratio is computed here, in millimetres, on the same test states and the same 64
execution-noise probes: D is the RMS spread of the three blocks' positions over the probes after
the 30-step hold. D_real comes from the Stage 0 simulator cache, D_pred from rolling the model
forward. A model that collapses every branch scores near 0; one that exaggerates scores above 1.

Only the injected error scales (0.5x, 1x, 2x) are used. The +-50 mm response sweep is not reported:
the training actions live far inside that range, so those curves are extrapolation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import RECORD_AT, SCALES  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_w5_eval import hold_index, load_model, predict  # noqa: E402

HOLD_AT = RECORD_AT.index(30)


def spread_mm(endings):
    """RMS spread of the block positions over probes. endings (probes, 9) in mm."""
    return float(np.sqrt(np.mean(np.sum((endings - endings.mean(0)) ** 2, 1))))


def separation(rows, cache_index, cache, model, scale, snippets, lmdb, sim_archive, device,
               reset_base):
    """Per state and error size: predicted and real ending spread in mm, plus the class."""
    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r
    replay = JengaReplay(lmdb)
    out = []
    with tempfile.TemporaryDirectory(prefix="jenga_w6_") as temp:
        sim = DirectJengaSim(str(extract_sim(sim_archive, temp)))
        try:
            for episode_id in sorted(wanted, key=int):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id) + reset_base)
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        start = step_state(sim)
                        index = cache_index[(episode_id, step)]
                        for s, scale_value in enumerate(SCALES):
                            windows = action_windows(episode.actions, step, snippets,
                                                     scale_value, own_hold=False)
                            predicted = 1000 * predict(model, scale, start, windows,
                                                       device)[:, hold_index(model, 30), 0:9]
                            real = 1000 * cache["pose"][index, s, :, HOLD_AT, 0:9]
                            out.append({
                                "episode_id": episode_id, "chunk_start": step,
                                "scale": str(scale_value),
                                "class": row["by_scale"][str(scale_value)]["class"],
                                "predicted_mm": spread_mm(predicted),
                                "real_mm": spread_mm(real)})
                    sim.execute(action)
        finally:
            sim.close()
            replay.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-report", default=None)
    ap.add_argument("--gate3", default=None)
    ap.add_argument("--test", default=str(ROOT / "results/jenga/holdout3_stage2_shared.json"))
    ap.add_argument("--test-stage0", default=str(ROOT / "results/jenga/holdout3_stage0.json"))
    ap.add_argument("--test-cache", default=str(ROOT / "results/jenga/holdout3_stage0_cache.npz"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--reset-seed-base", type=int, default=1000)
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(args.model, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    rows = json.loads(Path(args.test).read_text())["rows"]
    stage0 = json.loads(Path(args.test_stage0).read_text())["rows"]
    cache_index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0)}
    npz = np.load(args.test_cache, allow_pickle=False)
    cache = {"pose": npz["pose"]}

    records = separation(rows, cache_index, cache, model, scale, snippets, args.lmdb,
                         args.sim_archive, device, args.reset_seed_base)

    result = {"model": args.model, "protocol": {
        "separation": "RMS spread of the 3 block positions over 64 probes after the 30-step hold",
        "range": "injected execution error only (0.5x, 1x, 2x); no +-50 mm sweep"}}
    for s in map(str, SCALES):
        part = [r for r in records if r["scale"] == s]
        forks = [r for r in part if r["class"] == "topple_fork"]
        quiet = [r for r in part if r["class"] == "quiet"]
        entry = {"states": len(part), "fork_states": len(forks), "quiet_states": len(quiet)}
        for name, group in (("fork", forks), ("quiet", quiet)):
            if not group:
                continue
            predicted = float(np.sum([r["predicted_mm"] for r in group]))
            real = float(np.sum([r["real_mm"] for r in group]))
            entry[f"{name}_separation_ratio"] = predicted / max(real, 1e-9)
            entry[f"{name}_predicted_mm_median"] = float(
                np.median([r["predicted_mm"] for r in group]))
            entry[f"{name}_real_mm_median"] = float(np.median([r["real_mm"] for r in group]))
        # Contrast: how much more the endings spread at forks than at quiet states. The monitor
        # thresholds a single score, so it can only work if this ratio is well above 1, and the
        # honest target is the simulator's own value on the same states.
        if forks and quiet:
            entry["contrast_predicted"] = (entry["fork_predicted_mm_median"]
                                           / max(entry["quiet_predicted_mm_median"], 1e-9))
            entry["contrast_real"] = (entry["fork_real_mm_median"]
                                      / max(entry["quiet_real_mm_median"], 1e-9))
            entry["contrast_shortfall"] = entry["contrast_predicted"] / entry["contrast_real"]
        result[s] = entry

    if args.train_report and Path(args.train_report).exists():
        report = json.loads(Path(args.train_report).read_text())
        last = (report.get("rollout_history") or report["history"])[-1]["validation"]
        result["rollout_state_error"] = last["rollout_state_error"]
        result["validation_branch_ratio"] = last.get("branch_ratio")
    if args.gate3 and Path(args.gate3).exists():
        gate = json.loads(Path(args.gate3).read_text())["model"]
        result["gate3"] = {
            s: {"fork_recall": gate[s]["test_recall"]["rate"],
                "fork_recall_ci95": gate[s]["test_recall"]["ci95"],
                "non_fork_false_positive": gate[s]["test_false_alarms_quiet"]["rate"]}
            for s in gate}
    result["rows"] = records
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
