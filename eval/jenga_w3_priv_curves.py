"""Response curves for the privileged object-state head: is localisation representation-limited?

Same W0 states, directions and offset grid. The model sees the TRUE block poses at the chunk start
instead of DINO patches, and predicts codes over TRUE ending poses, so its curve is measured in
the same physical units as reality's. If the fork/quiet jump contrast approaches reality's 4.14,
perception is the binding constraint and an object-centric model is worth building; if it stays
near 1.0, object features will not fix localisation either.
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
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_state_data import full_state  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_w0_response_curves import (DIRECTIONS, OFFSETS_MM, curve_metrics,  # noqa: E402
                                      offset_windows, simulate_curve)
from jenga_w2_discrete import EndingHead, HOLD_STEPS, action_features  # noqa: E402


def load_head(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = EndingHead(int(state["input_dim"]), int(state["action_dim"]),
                       max(v["codes"] for v in state["sizes"].values())).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--states", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--head", action="append", required=True, help="name=checkpoint.pt")
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w3_privileged_curves.json"))
    ap.add_argument("--forks", type=int, default=20)
    ap.add_argument("--quiet", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = {}
    for spec in args.head:
        name, path = spec.split("=", 1)
        heads[name] = load_head(path, device)

    rows = json.loads(Path(args.states).read_text())["rows"]
    forks = [r for r in rows if r["by_scale"]["2.0"]["class"] == "topple_fork"][:args.forks]
    quiet = [r for r in rows if r["by_scale"]["2.0"]["class"] == "quiet"][:args.quiet]
    states = [dict(r, group=g) for g, part in (("fork", forks), ("quiet", quiet)) for r in part]
    by_episode = {}
    for r in states:
        by_episode.setdefault(r["episode_id"], []).append(r)

    replay = JengaReplay(args.lmdb)
    records = []
    with tempfile.TemporaryDirectory(prefix="jenga_w3c_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(by_episode, key=int)):
                episode = replay.episode(episode_id)
                starts = {r["chunk_start"]: r for r in by_episode[episode_id]}
                sim.reset(int(episode_id))
                for step, action in enumerate(episode.actions):
                    if step in starts:
                        # Each head gets the state it was trained on: 27 = blocks only,
                        # 70 = full state (blocks, relative geometry, gripper, velocities,
                        # contacts).
                        state_by_dim = {27: all_block_pose(sim), 70: full_state(sim)}
                        chunk = episode.actions[step:step + HORIZON]
                        snapshot = sim.snapshot()
                        for axis, direction in DIRECTIONS.items():
                            positions, topples = simulate_curve(sim, snapshot, chunk, direction,
                                                                OFFSETS_MM)
                            windows = offset_windows(episode.actions, step, direction, OFFSETS_MM)
                            action_input = action_features(
                                torch.from_numpy(np.asarray(windows, np.float32)))
                            record = {"episode_id": episode_id, "chunk_start": step,
                                      "group": starts[step]["group"], "direction": axis,
                                      "real": curve_metrics(positions, OFFSETS_MM),
                                      "real_topple_count": int(sum(topples)), "models": {}}
                            for name, (head, state) in heads.items():
                                start_pose = state_by_dim[int(state["input_dim"])]
                                normalised = (start_pose - state["input_mean"]) / state[
                                    "input_scale"]
                                batch = torch.from_numpy(
                                    np.tile(normalised, (len(OFFSETS_MM), 1)).astype(np.float32))
                                with torch.inference_mode():
                                    logits = head(batch.to(device), action_input.to(device))
                                entry = {}
                                for slot, held in enumerate(HOLD_STEPS):
                                    probabilities = logits[slot].softmax(1)
                                    centres = torch.as_tensor(state["codebooks"][held],
                                                              device=device, dtype=torch.float32)
                                    embedding = (probabilities @ centres).cpu().numpy()
                                    codes = probabilities.argmax(1).cpu().numpy()
                                    metrics = curve_metrics(embedding, OFFSETS_MM)
                                    metrics["code_switches"] = int(np.sum(np.diff(codes) != 0))
                                    entry[f"hold{held}"] = metrics
                                record["models"][name] = entry
                            records.append(record)
                    sim.execute(action)
                print(f"  w3 curves {number + 1}/{len(by_episode)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()

    def median_of(values, key):
        """Flat curves have an undefined jump ratio; report how many and ignore them."""
        present = [v[key] for v in values if v[key] is not None]
        return (float(np.median(present)) if present else None,
                len(values) - len(present))

    summary, gates = {}, {}
    for group in ("fork", "quiet"):
        part = [r for r in records if r["group"] == group]
        real_jump, real_flat = median_of([r["real"] for r in part], "jump_ratio")
        entry = {"curves": len(part), "real_jump_ratio_median": real_jump,
                 "real_flat_curves": real_flat}
        for name in heads:
            for held in HOLD_STEPS:
                values = [r["models"][name][f"hold{held}"] for r in part]
                jump, degenerate = median_of(values, "jump_ratio")
                entry[f"{name}_hold{held}"] = {
                    "jump_ratio_median": jump, "flat_curves": degenerate,
                    "spread_median": float(np.median([v["spread"] for v in values])),
                    "code_switches_median": float(np.median([v["code_switches"]
                                                             for v in values]))}
        summary[group] = entry
    for name in heads:
        for held in HOLD_STEPS:
            key = f"{name}_hold{held}"
            fork, quiet_entry = summary["fork"][key], summary["quiet"][key]
            both = fork["jump_ratio_median"] is not None and (
                quiet_entry["jump_ratio_median"] is not None)
            gates[key] = {
                "fork_over_quiet_jump_ratio": (
                    fork["jump_ratio_median"] / max(quiet_entry["jump_ratio_median"], 1e-9)
                    if both else None),
                "fork_over_quiet_spread": fork["spread_median"] / max(
                    quiet_entry["spread_median"], 1e-9),
                "flat_curves_fork": fork["flat_curves"],
                "flat_curves_quiet": quiet_entry["flat_curves"]}
    gates["reality"] = {"fork_over_quiet_jump_ratio": summary["fork"][
        "real_jump_ratio_median"] / max(summary["quiet"]["real_jump_ratio_median"], 1e-9)}
    result = {"protocol": {"input": "TRUE block poses at the chunk start (privileged)",
                           "target": "codes over TRUE ending poses",
                           "upper_bound": "a camera cannot supply this input"},
              "summary": summary, "gates": gates, "records": records}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
