"""Gate 3 for a W5 model: persistent-branch recall at a fixed safe-control false-alarm budget.

PLAN_WORLDMODEL W5 step 2, following the external plan's Phase 4: the operating point is set on
SAFE controls only -- no failure labels -- on development data, then evaluated once on test data.

  score      spread of the model's predicted endings over the same 64 execution-noise probes
             (block positions after the 30-step hold, mm)
  operating  per error size, the 95th percentile of the score over DEVELOPMENT quiet states
  point      (batch 1: unanimous no-topple, no physical split), i.e. <= 5% false alarms there
  test       applied once to batch 2: recall on topple forks, false alarms on quiet states, and
             separately on non-topple physical forks (reported, never used to set the threshold)
  intervals  95% bootstrap over whole episodes, because states share episodes

Reference: the identical score, calibration and test on REAL rendered endings (DINO latent spread
from the Stage 2 runs), so the model's number has a like-for-like ceiling.

Caveat: batches 1 and 2 are distinct states from the same pool of 57 held-out episodes, so
development and test share episodes. The threshold uses only the quiet-state score distribution.
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
from jenga_stage0_noise_oracle import SCALES  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_w5_eval import HOLD_INDEX, load_model, predict  # noqa: E402

BUDGET = 0.05
NEGATIVE_PHYSICAL = ("nudge_fork", "small_split")


def model_scores(rows, model, scale, snippets, lmdb, sim_archive, device):
    """Predicted ending spread (mm) per state and error size, plus class and episode."""
    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r
    replay = JengaReplay(lmdb)
    out = []
    with tempfile.TemporaryDirectory(prefix="jenga_gate3_") as temp:
        sim = DirectJengaSim(str(extract_sim(sim_archive, temp)))
        try:
            for episode_id in sorted(wanted, key=int):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id))
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        start = step_state(sim)
                        for s in SCALES:
                            windows = action_windows(episode.actions, step, snippets, s,
                                                     own_hold=False)
                            ends = 1000 * predict(model, scale, start, windows,
                                                  device)[:, HOLD_INDEX[30], 0:9]
                            spread = float(np.sqrt(np.mean(np.sum(
                                (ends - ends.mean(0)) ** 2, 1))))
                            out.append({"episode_id": episode_id, "chunk_start": step,
                                        "scale": str(s), "score": spread,
                                        "class": row["by_scale"][str(s)]["class"]})
                    sim.execute(action)
        finally:
            sim.close()
            replay.close()
    return out


def real_scores(rows):
    """The same score on REAL rendered endings: DINO latent spread from the Stage 2 run."""
    return [{"episode_id": r["episode_id"], "chunk_start": r["chunk_start"], "scale": s,
             "score": b["visual"]["latent_spread_rms"], "class": b["class"]}
            for r in rows for s, b in r["by_scale"].items()]


def clustered_rate(records, key, n_boot=2000, seed=0):
    """Point estimate and 95% interval of mean(key), resampling whole episodes."""
    if not records:
        return None
    episodes = sorted({r["episode_id"] for r in records})
    by_episode = {e: [r[key] for r in records if r["episode_id"] == e] for e in episodes}
    point = float(np.mean([r[key] for r in records]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(episodes, len(episodes), replace=True)
        values = [v for e in draw for v in by_episode[e]]
        boots.append(np.mean(values))
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"rate": point, "ci95": [float(low), float(high)], "n": len(records),
            "episodes": len(episodes)}


def gate3(dev, test):
    result = {}
    for s in map(str, SCALES):
        dev_quiet = [r["score"] for r in dev if r["scale"] == s and r["class"] == "quiet"]
        threshold = float(np.quantile(dev_quiet, 1 - BUDGET, method="higher"))
        part = [dict(r, alarm=r["score"] > threshold) for r in test if r["scale"] == s]
        forks = [r for r in part if r["class"] == "topple_fork"]
        quiet = [r for r in part if r["class"] == "quiet"]
        physical = [r for r in part if r["class"] in NEGATIVE_PHYSICAL]
        dev_fa = float(np.mean([v > threshold for v in dev_quiet]))
        result[s] = {"threshold": threshold, "dev_quiet_states": len(dev_quiet),
                     "dev_false_alarms": dev_fa,
                     "test_recall": clustered_rate(forks, "alarm"),
                     "test_false_alarms_quiet": clustered_rate(quiet, "alarm"),
                     "test_alarms_on_nontopple_physical_forks": clustered_rate(physical, "alarm")}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--model", default=str(ROOT / "results/jenga/w5_stepnet.pt"))
    ap.add_argument("--dev", default=str(ROOT / "results/jenga/holdout_stage2_shared.json"))
    ap.add_argument("--test", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w5_gate3.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(args.model, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    dev_rows = json.loads(Path(args.dev).read_text())["rows"]
    test_rows = json.loads(Path(args.test).read_text())["rows"]

    model_dev = model_scores(dev_rows, model, scale, snippets, args.lmdb, args.sim_archive, device)
    model_test = model_scores(test_rows, model, scale, snippets, args.lmdb, args.sim_archive,
                              device)
    result = {"protocol": {"score": "spread of predicted endings over 64 execution-noise probes",
                           "operating_point": f"{int(100 * (1 - BUDGET))}th percentile of "
                                              "development QUIET-state scores, per error size",
                           "development": args.dev, "test": args.test,
                           "intervals": "95% bootstrap over whole episodes",
                           "labels_used_for_threshold": "none (quiet class only)",
                           "caveat": "dev and test are distinct states from the same episode pool"},
              "model": gate3(model_dev, model_test),
              "real_endings_reference": gate3(real_scores(dev_rows), real_scores(test_rows))}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")

    def fmt(entry):
        return "-" if entry is None else (f"{100 * entry['rate']:.0f}% "
                                          f"[{100 * entry['ci95'][0]:.0f}-"
                                          f"{100 * entry['ci95'][1]:.0f}] n={entry['n']}")
    for name in ("model", "real_endings_reference"):
        print(name)
        for s, e in result[name].items():
            print(f"  {s}x  recall {fmt(e['test_recall'])}   FA quiet "
                  f"{fmt(e['test_false_alarms_quiet'])}   alarms on non-topple physical forks "
                  f"{fmt(e['test_alarms_on_nontopple_physical_forks'])}")


if __name__ == "__main__":
    main()
