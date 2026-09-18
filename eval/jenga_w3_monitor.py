"""The monitor end to end on PREDICTED endings from the full-state model.

The full-state privileged model cannot localise a boundary (fork/quiet jump contrast 0.88-0.99
against reality's 4.17) but its response is 8-9x larger at fork states than at quiet ones. The
monitor's question is a magnitude question -- could nearby executions end differently? -- so this
runs the real thing: the same 64 execution-noise probes used everywhere in Stages 0-3, the model's
predicted endings, and the frozen fork rules, on the same holdout states as the real-ending
benchmark (88% recall at 1% false alarms with a shared hold).

It is not deployable as it stands: the input is simulator state. If it works, the open problem
narrows from "make a world model that jumps" to "recover ~70 dimensions of object state from
images", which is an ordinary perception problem.
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
from outcome_modes import coarse_persistent_fork, groupings_persist, multi_mode_test  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import SCALES  # noqa: E402
from jenga_stage2_visual_forks import fork_test  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import full_state  # noqa: E402
from jenga_w2_discrete import HOLD_STEPS, action_features  # noqa: E402
from jenga_w3_priv_curves import load_head  # noqa: E402


def predicted_endings(head, state, start_state, windows, device):
    """(probes, holds, code-embedding dims): the mean embedding of each predicted distribution."""
    normalised = (start_state - state["input_mean"]) / state["input_scale"]
    batch = torch.from_numpy(np.tile(normalised, (len(windows), 1)).astype(np.float32))
    action_input = action_features(torch.from_numpy(np.asarray(windows, np.float32)))
    with torch.inference_mode():
        logits = head(batch.to(device), action_input.to(device))
    out = []
    for slot, held in enumerate(HOLD_STEPS):
        centres = torch.as_tensor(state["codebooks"][held], device=device, dtype=torch.float32)
        out.append((logits[slot].softmax(1) @ centres).cpu().numpy())
    return np.stack(out)  # (holds, probes, dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--states", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--head", default=str(ROOT / "results/jenga/w3_fullstate_head.pt"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w3_monitor.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, state = load_head(args.head, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    rows = json.loads(Path(args.states).read_text())["rows"]
    wanted = {}
    for row in rows:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row

    replay = JengaReplay(args.lmdb)
    results = []
    with tempfile.TemporaryDirectory(prefix="jenga_w3m_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(wanted, key=int)):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id))
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        start_state = full_state(sim)
                        entry = {"episode_id": episode_id, "chunk_start": step, "by_scale": {}}
                        for s, scale in enumerate(SCALES):
                            windows = action_windows(episode.actions, step, snippets, scale,
                                                     own_hold=False)
                            endings = predicted_endings(head, state, start_state, windows, device)
                            test = fork_test(endings)
                            early = multi_mode_test(endings[0])
                            late = multi_mode_test(endings[1])
                            both = early["groups"] >= 2 and late["groups"] >= 2
                            spread = float(np.sqrt(np.mean(np.sum(
                                (endings[1] - endings[1].mean(0)) ** 2, 1))))
                            entry["by_scale"][str(scale)] = {
                                "class": row["by_scale"][str(scale)]["class"],
                                "topple_count": row["by_scale"][str(scale)]["topple_count"],
                                "old_pc1_alarm": test["alarm"],
                                "revised_alarm": bool(both and coarse_persistent_fork(
                                    early["labels"], late["labels"])),
                                "dominant_alarm": bool(both and groupings_persist(
                                    early["dominant"], late["dominant"])),
                                "predicted_spread": spread,
                                "latent_separation": test["latent_separation"]}
                        results.append(entry)
                    sim.execute(action)
                print(f"  w3 monitor {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()

    def auc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l]
        neg = [s for s, l in zip(scores, labels) if not l]
        if not pos or not neg:
            return None
        wins = sum((p > n) + .5 * (p == n) for p in pos for n in neg)
        return float(wins / (len(pos) * len(neg)))

    summary = {}
    for scale in map(str, SCALES):
        part = [r["by_scale"][scale] for r in results]
        forks = [p for p in part if p["class"] == "topple_fork"]
        quiet = [p for p in part if p["class"] == "quiet"]
        entry = {"fork_states": len(forks), "quiet_states": len(quiet)}
        for rule in ("old_pc1_alarm", "revised_alarm", "dominant_alarm"):
            entry[rule] = {"recall": (sum(f[rule] for f in forks) / len(forks)) if forks else None,
                           "false_alarms": (sum(q[rule] for q in quiet) / len(quiet))
                           if quiet else None}
        graded = forks + quiet
        entry["spread_auc"] = auc([g["predicted_spread"] for g in graded],
                                  [g["class"] == "topple_fork" for g in graded])
        entry["separation_auc"] = auc([g["latent_separation"] for g in graded],
                                      [g["class"] == "topple_fork" for g in graded])
        summary[scale] = entry

    Path(args.output).write_text(json.dumps(
        {"protocol": {"input": "privileged full simulator state; NOT deployable as it stands",
                      "probes": len(snippets), "rules": "frozen Stage 2/2b fork rules",
                      "reference": "real endings, shared hold: 88% recall at 1% false alarms",
                      "spread_auc": "ranking fork vs quiet states by predicted ending spread"},
         "summary": summary, "rows": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
