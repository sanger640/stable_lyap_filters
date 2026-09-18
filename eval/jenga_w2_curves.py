"""Grade the W2a discrete ending head on the W0 response-curve benchmark.

Same states, directions and signed offset grid as eval/jenga_w0_response_curves.py. The model's
"ending" is the mean embedding of its predicted code distribution, so spread and jump ratio are
computed exactly as for a continuous model. The gate is the fork-state jump ratio against the same
model's quiet-state jump ratio (reality: 14.9 vs 3.6), plus the fork/quiet spread ratio
(reality 1.81).

Also reports argmax code agreement across the grid: the fraction of adjacent offsets whose most
likely ending code differs, which is the discrete analogue of a jump and cannot be faked by a ramp.
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
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           NUM_HIST, load_world_model, normalise_proprio, preprocess_frames)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_w0_response_curves import (DIRECTIONS, OFFSETS_MM, curve_metrics,  # noqa: E402
                                      offset_windows, simulate_curve)
from jenga_w2_discrete import EndingHead, HOLD_STEPS, action_features  # noqa: E402


def load_head(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = EndingHead(int(state["history_dim"]), int(state["action_dim"]),
                       max(v["codes"] for v in state["sizes"].values())).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def encode_history(world_model, device, frames, proprio):
    with torch.inference_mode():
        z = world_model.encode_obs({
            "visual": preprocess_frames(frames[:, None], device),
            "proprio": normalise_proprio(proprio[:, None], device)})["visual"][:, 0]
    return z.flatten(1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--states", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--head", action="append", required=True, help="name=checkpoint.pt")
    ap.add_argument("--encoder", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w2_response_curves.json"))
    ap.add_argument("--forks", type=int, default=20)
    ap.add_argument("--quiet", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_model = load_world_model(args.encoder, device)  # frozen encoder only
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
    with tempfile.TemporaryDirectory(prefix="jenga_w2c_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(by_episode, key=int)):
                episode = replay.episode(episode_id)
                starts = {r["chunk_start"]: r for r in by_episode[episode_id]}
                sim.reset(int(episode_id))
                frames, props = [sim.render()], [sim.proprio()]
                for step, action in enumerate(episode.actions):
                    if step in starts:
                        history_latents = encode_history(
                            world_model, device, np.stack(frames[-NUM_HIST:]),
                            np.stack(props[-NUM_HIST:]))
                        chunk = episode.actions[step:step + HORIZON]
                        snapshot = sim.snapshot()
                        for axis, direction in DIRECTIONS.items():
                            positions, topples = simulate_curve(sim, snapshot, chunk, direction,
                                                                OFFSETS_MM)
                            windows = offset_windows(episode.actions, step, direction, OFFSETS_MM)
                            record = {"episode_id": episode_id, "chunk_start": step,
                                      "group": starts[step]["group"], "direction": axis,
                                      "real": curve_metrics(positions, OFFSETS_MM),
                                      "real_topple_count": int(sum(topples)), "models": {}}
                            action_input = action_features(
                                torch.from_numpy(np.asarray(windows, np.float32)))
                            for name, (head, state) in heads.items():
                                mean_h, comp_h = state["history_projection"]
                                projected = np.concatenate(
                                    [(history_latents[i] - mean_h) @ comp_h.T
                                     for i in range(NUM_HIST)])
                                batch = torch.from_numpy(
                                    np.tile(projected, (len(OFFSETS_MM), 1)).astype(np.float32))
                                with torch.inference_mode():
                                    logits = head(batch.to(device), action_input.to(device))
                                entry = {}
                                for slot, held in enumerate(HOLD_STEPS):
                                    size = state["sizes"][held]["codes"]
                                    probabilities = logits[slot][:, :size].softmax(1)
                                    centres = torch.as_tensor(state["codebooks"][held],
                                                              device=device, dtype=torch.float32)
                                    embedding = (probabilities @ centres).cpu().numpy()
                                    codes = probabilities.argmax(1).cpu().numpy()
                                    metrics = curve_metrics(embedding, OFFSETS_MM)
                                    metrics["code_switches"] = int(np.sum(np.diff(codes) != 0))
                                    metrics["distinct_codes"] = int(len(np.unique(codes)))
                                    entry[f"hold{held}"] = metrics
                                record["models"][name] = entry
                            records.append(record)
                    sim.execute(action)
                    frames.append(sim.render()); props.append(sim.proprio())
                    frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
                print(f"  w2 curves {number + 1}/{len(by_episode)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()

    summary = {}
    for group in ("fork", "quiet"):
        part = [r for r in records if r["group"] == group]
        entry = {"curves": len(part),
                 "real_jump_ratio_median": float(np.median([r["real"]["jump_ratio"]
                                                            for r in part])),
                 "real_spread_median": float(np.median([r["real"]["spread"] for r in part]))}
        for name in heads:
            for held in HOLD_STEPS:
                values = [r["models"][name][f"hold{held}"] for r in part]
                entry[f"{name}_hold{held}"] = {
                    "jump_ratio_median": float(np.median([v["jump_ratio"] for v in values])),
                    "spread_median": float(np.median([v["spread"] for v in values])),
                    "code_switches_median": float(np.median([v["code_switches"] for v in values])),
                    "distinct_codes_median": float(np.median([v["distinct_codes"]
                                                              for v in values]))}
        summary[group] = entry
    gates = {}
    for name in heads:
        for held in HOLD_STEPS:
            key = f"{name}_hold{held}"
            fork, quiet_entry = summary["fork"][key], summary["quiet"][key]
            gates[key] = {
                "fork_over_quiet_jump_ratio": fork["jump_ratio_median"] / max(
                    quiet_entry["jump_ratio_median"], 1e-9),
                "fork_over_quiet_spread": fork["spread_median"] / max(
                    quiet_entry["spread_median"], 1e-9),
                "gate_jump_ge_3": bool(fork["jump_ratio_median"]
                                       >= 3 * quiet_entry["jump_ratio_median"]),
                "gate_spread_ge_1_5": bool(fork["spread_median"]
                                           >= 1.5 * quiet_entry["spread_median"])}
    result = {"protocol": {"benchmark": "W0 states, directions and offsets",
                           "model_ending": "mean embedding of the predicted code distribution",
                           "reality_reference": {"fork_over_quiet_jump_ratio": 14.9 / 3.6,
                                                 "fork_over_quiet_spread": 1.81},
                           "gate": "fork jump ratio >= 3x the same model's quiet jump ratio, and "
                                   "fork/quiet spread >= 1.5"},
              "summary": summary, "gates": gates, "records": records}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
