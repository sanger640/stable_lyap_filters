"""Grade the W5 step-wise dynamics model: W0 response curves, then the monitor end to end.

The model predicts block state directly, so its response curves are in the SAME millimetres as
reality's (block positions after the hold) -- the first like-for-like comparison of spread as well
as of jump ratio. Everything else is fixed in advance:

  W0 gate: fork jump ratio >= 3x the same model's quiet value (reality 4.17), fork/quiet
           spread >= 1.5 (reality 1.81).
  Monitor: the 64 execution-noise probes and the frozen fork rules on predicted endings, against
           the real-ending reference of 88% recall at 1% false alarms (shared hold).

Also reports the most direct physical check: the fraction of probes at each state the model
predicts will topple a neighbour (tilt >= 45 degrees, read from the predicted rotation), against
the simulator's own count. Topple labels are used only for grading.
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
from state_dynamics import StepGraphMoE, StepGraphNet, rollout  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import SCALES  # noqa: E402
from jenga_stage2_visual_forks import fork_test  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_w0_response_curves import (DIRECTIONS, OFFSETS_MM, curve_metrics,  # noqa: E402
                                      offset_windows, simulate_curve)


def hold_index(model, held):
    """Rollout index of the reading after `held` hold steps, at the model's resolution."""
    return (HORIZON + held) * getattr(model, "substeps", 1) - 1


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind", "single") == "moe":
        model = StepGraphMoE(state["hidden"], state["rounds"], state["experts"]).to(device)
    else:
        model = StepGraphNet(state["hidden"], state["rounds"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    model.substeps = int(state.get("substeps", 1))
    scale = (state["block_scale"].to(device), state["grip_scale"].to(device))
    return model, scale


def predict(model, scale, start, windows, device):
    """start (61,), windows (P, 40, 4) -> predicted states (P, 38 x sub-steps, 61)."""
    state = torch.as_tensor(np.tile(start, (len(windows), 1)), dtype=torch.float32, device=device)
    held = np.repeat(np.asarray(windows)[:, 2:], getattr(model, "substeps", 1), axis=1)
    actions = torch.as_tensor(held, dtype=torch.float32, device=device)
    with torch.no_grad():
        return rollout(model, state, actions, scale).cpu().numpy()


def neighbour_tilt_deg(states):
    """Tilt of blocks 1 and 2 from the predicted rotation: arccos of the body z-axis's z.

    all_block_pose stores xmat[:, :, :2] flattened ROW-major, so the six numbers per block are
    (r00, r01, r10, r11, r20, r21): column 0 is the even entries, column 1 the odd ones. Reading
    them as two contiguous columns reports every block as ~90 degrees tilted.
    """
    tilts = []
    for b in (1, 2):
        block = states[..., 9 + 6 * b: 15 + 6 * b]
        c0, c1 = block[..., 0::2], block[..., 1::2]
        c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True).clip(1e-9)
        c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True).clip(1e-9)
        z = np.cross(c0, c1)[..., 2]
        tilts.append(np.degrees(np.arccos(np.clip(z, -1, 1))))
    return np.stack(tilts, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--states", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--model", default=str(ROOT / "results/jenga/w5_stepnet.pt"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w5_eval.json"))
    ap.add_argument("--forks", type=int, default=20)
    ap.add_argument("--quiet", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(args.model, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    rows = json.loads(Path(args.states).read_text())["rows"]
    curve_forks = {(r["episode_id"], r["chunk_start"]) for r in rows
                   if r["by_scale"]["2.0"]["class"] == "topple_fork"}
    curve_quiet = {(r["episode_id"], r["chunk_start"]) for r in rows
                   if r["by_scale"]["2.0"]["class"] == "quiet"}
    curve_forks = set(sorted(curve_forks, key=lambda k: (int(k[0]), k[1]))[:args.forks])
    curve_quiet = set(sorted(curve_quiet, key=lambda k: (int(k[0]), k[1]))[:args.quiet])
    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r

    replay = JengaReplay(args.lmdb)
    curves, monitor = [], []
    with tempfile.TemporaryDirectory(prefix="jenga_w5_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(wanted, key=int)):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id))
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        start = step_state(sim)
                        key = (episode_id, step)
                        # --- W0 response curves, for the benchmark subset only.
                        if key in curve_forks or key in curve_quiet:
                            chunk = episode.actions[step:step + HORIZON]
                            snapshot = sim.snapshot()
                            for axis, direction in DIRECTIONS.items():
                                positions, _ = simulate_curve(sim, snapshot, chunk, direction,
                                                              OFFSETS_MM)
                                windows = offset_windows(episode.actions, step, direction,
                                                         OFFSETS_MM)
                                predicted = predict(model, scale, start, windows, device)
                                model_pos = 1000 * predicted[:, hold_index(model, 30), 0:9]
                                curves.append({
                                    "episode_id": episode_id, "chunk_start": step,
                                    "group": "fork" if key in curve_forks else "quiet",
                                    "direction": axis,
                                    "real": curve_metrics(positions, OFFSETS_MM),
                                    "model": curve_metrics(model_pos, OFFSETS_MM)})
                        # --- The monitor end to end, every holdout state.
                        entry = {"episode_id": episode_id, "chunk_start": step, "by_scale": {}}
                        for scale_value in SCALES:
                            windows = action_windows(episode.actions, step, snippets,
                                                     scale_value, own_hold=False)
                            predicted = predict(model, scale, start, windows, device)
                            endings = np.stack([1000 * predicted[:, hold_index(model, h), 0:9]
                                                for h in (10, 30)])
                            tilt = neighbour_tilt_deg(predicted[:, hold_index(model, 30)])
                            # Count only NEW topples, as the Stage 0 answer key does: a neighbour
                            # already down at the chunk start is not a topple caused by this chunk.
                            eligible = neighbour_tilt_deg(start[None])[0] < TOPPLE_DEG
                            test = fork_test(endings)
                            early, late = multi_mode_test(endings[0]), multi_mode_test(endings[1])
                            both = early["groups"] >= 2 and late["groups"] >= 2
                            spread = float(np.sqrt(np.mean(np.sum(
                                (endings[1] - endings[1].mean(0)) ** 2, 1))))
                            truth = row["by_scale"][str(scale_value)]
                            entry["by_scale"][str(scale_value)] = {
                                "class": truth["class"], "true_topples": truth["topple_count"],
                                "predicted_topples": int(np.sum(np.any(
                                    (tilt >= TOPPLE_DEG) & eligible[None], 1))),
                                "old_pc1_alarm": test["alarm"],
                                "revised_alarm": bool(both and coarse_persistent_fork(
                                    early["labels"], late["labels"])),
                                "dominant_alarm": bool(both and groupings_persist(
                                    early["dominant"], late["dominant"])),
                                "predicted_spread_mm": spread}
                        monitor.append(entry)
                    sim.execute(action)
                print(f"  w5 eval {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()

    def median(values):
        present = [v for v in values if v is not None]
        return float(np.median(present)) if present else None

    w0 = {}
    for group in ("fork", "quiet"):
        part = [c for c in curves if c["group"] == group]
        w0[group] = {"curves": len(part),
                     "real_jump": median([c["real"]["jump_ratio"] for c in part]),
                     "model_jump": median([c["model"]["jump_ratio"] for c in part]),
                     "real_spread_mm": median([c["real"]["spread"] for c in part]),
                     "model_spread_mm": median([c["model"]["spread"] for c in part])}
    gate = {"model_fork_over_quiet_jump": w0["fork"]["model_jump"] / w0["quiet"]["model_jump"],
            "real_fork_over_quiet_jump": w0["fork"]["real_jump"] / w0["quiet"]["real_jump"],
            "model_fork_over_quiet_spread": w0["fork"]["model_spread_mm"]
            / w0["quiet"]["model_spread_mm"],
            "real_fork_over_quiet_spread": w0["fork"]["real_spread_mm"]
            / w0["quiet"]["real_spread_mm"]}
    gate["passes_jump_gate"] = bool(gate["model_fork_over_quiet_jump"] >= 3)
    gate["passes_spread_gate"] = bool(gate["model_fork_over_quiet_spread"] >= 1.5)

    def auc(scores, labels):
        pos = [s for s, lab in zip(scores, labels) if lab]
        neg = [s for s, lab in zip(scores, labels) if not lab]
        if not pos or not neg:
            return None
        return float(sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))

    monitor_summary = {}
    for scale_value in map(str, SCALES):
        part = [m["by_scale"][scale_value] for m in monitor]
        forks = [p for p in part if p["class"] == "topple_fork"]
        quiet = [p for p in part if p["class"] == "quiet"]
        entry = {"fork_states": len(forks), "quiet_states": len(quiet)}
        for rule in ("old_pc1_alarm", "revised_alarm", "dominant_alarm"):
            entry[rule] = {"recall": sum(f[rule] for f in forks) / len(forks) if forks else None,
                           "false_alarms": sum(q[rule] for q in quiet) / len(quiet)
                           if quiet else None}
        graded = forks + quiet
        entry["spread_auc"] = auc([g["predicted_spread_mm"] for g in graded],
                                  [g["class"] == "topple_fork" for g in graded])
        entry["forks_where_model_predicts_any_topple"] = (
            sum(f["predicted_topples"] > 0 for f in forks) / len(forks) if forks else None)
        entry["quiet_where_model_predicts_any_topple"] = (
            sum(q["predicted_topples"] > 0 for q in quiet) / len(quiet) if quiet else None)
        monitor_summary[scale_value] = entry

    result = {"protocol": {"model": args.model, "input": "privileged state (oracle variant)",
                           "curve_units": "block positions in mm after the 30-step hold, same "
                                          "units as reality", "reference": "real endings: 88% "
                                          "recall at 1% false alarms (shared hold)"},
              "w0": w0, "w0_gate": gate, "monitor": monitor_summary,
              "curves": curves, "monitor_rows": monitor}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"w0": w0, "w0_gate": gate, "monitor": monitor_summary}, indent=1))


if __name__ == "__main__":
    main()
