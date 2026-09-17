"""Stage 3: the fork test on WORLD-MODEL PREDICTED endings (the deployable input).

Identical to Stage 2 except that nothing after the chunk is simulated. The three real frames at
the chunk start warm-start the bundled world model, each of the 64 noise-perturbed action windows
is rolled out H=8 plus a 30-step hold, and the predicted latents after 10 and 30 held steps are
read. Everything else -- probes, scales, per-state exact PCA, the frozen rules -- is unchanged.

Physical classes and topple counts come from the matching Stage 2 run (the same states and the
same probes, simulated) and are used only for grading. Output shape matches Stage 2 so
`eval/jenga_stage2b_multimode.py` can score every rule on it.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           NUM_HIST, load_world_model, normalise_actions, normalise_proprio,
                           preprocess_frames)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD, SCALES  # noqa: E402
from jenga_stage2_visual_forks import FRAME_AT, fork_test  # noqa: E402

# rollout index: NUM_HIST observed frames then 2 history + H + hold transitions
PRED_INDEX = {held: 2 + HORIZON + held for held in FRAME_AT}


def action_windows(actions, start, snippets, scale, own_hold):
    """(probes, 2 + H + HOLD, 4) absolute-target windows, gripper never perturbed."""
    history = np.asarray(actions[start - 2:start], np.float32)
    chunk = np.asarray(actions[start:start + HORIZON], np.float32)
    windows = []
    for noise in snippets:
        probe = chunk.copy()
        probe[:, :3] -= noise * scale
        hold = probe[-1] if own_hold else chunk[-1]
        windows.append(np.concatenate([history, probe, np.repeat(hold[None], HOLD, axis=0)]))
    return np.stack(windows)


def predict_state(model, device, frames, proprio, windows, batch_size=32):
    """Predicted latents at the recorded held steps, per probe."""
    initial_visual = preprocess_frames(frames[None], device)
    initial_proprio = normalise_proprio(proprio[None], device)
    out = {held: [] for held in FRAME_AT}
    for first in range(0, len(windows), batch_size):
        batch = windows[first:first + batch_size]
        obs = {"visual": initial_visual.expand(len(batch), -1, -1, -1, -1),
               "proprio": initial_proprio.expand(len(batch), -1, -1)}
        with torch.inference_mode():
            prediction, _ = model.rollout(obs, normalise_actions(batch, device))
        visual = prediction["visual"]
        if not torch.isfinite(visual).all():
            raise RuntimeError("rollout produced a non-finite latent")
        for held, index in PRED_INDEX.items():
            out[held].append(visual[:, index].float().flatten(1))
    return {held: torch.cat(v) for held, v in out.items()}


def exact_coordinates(tokens):
    x = tokens.double()
    x = x - x.mean(0)
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    return (u * s).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--stage2", required=True, help="matching Stage 2 result JSON (classes)")
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--own-hold", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N states only (timing check)")
    args = ap.parse_args()

    stage2 = json.loads(Path(args.stage2).read_text())
    stage2_rows = stage2["rows"][:args.limit] if args.limit else stage2["rows"]
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    device = "cuda"
    model = load_world_model(args.checkpoint, device)
    replay = JengaReplay(args.lmdb)
    wanted = {}
    for row in stage2_rows:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    coords = np.empty((len(stage2_rows), len(SCALES), len(FRAME_AT), len(snippets),
                       len(snippets)), np.float32)
    index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage2_rows)}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="jenga_stage3_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(wanted, key=int)):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id))
                frames, props = [sim.render()], [sim.proprio()]
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        i = index[(episode_id, step)]
                        history_frames = np.stack(frames[-NUM_HIST:])
                        history_proprio = np.stack(props[-NUM_HIST:])
                        for s, scale in enumerate(SCALES):
                            windows = action_windows(episode.actions, step, snippets, scale,
                                                     args.own_hold)
                            tokens = predict_state(model, device, history_frames,
                                                   history_proprio, windows)
                            for t, held in enumerate(FRAME_AT):
                                coords[i, s, t] = exact_coordinates(tokens[held])
                    sim.execute(action)
                    frames.append(sim.render()); props.append(sim.proprio())
                    frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
                print(f"  stage3 {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()
    np.savez_compressed(args.cache, coords=coords)

    rows = []
    for i, prow in enumerate(stage2_rows):
        row = {"episode_id": prow["episode_id"], "chunk_start": prow["chunk_start"],
               "stratum": prow["stratum"], "by_scale": {}}
        for s, scale in enumerate(prow["by_scale"]):
            b = prow["by_scale"][scale]
            row["by_scale"][scale] = {"class": b["class"], "topple_count": b["topple_count"],
                                      "arm_end_spread_mm": b["arm_end_spread_mm"],
                                      "visual": fork_test(coords[i, s])}
        rows.append(row)
    result = {"protocol": {
        "input": "world-model PREDICTED latents after 10 and 30 held steps, warm-started from "
                 "the three real frames at the chunk start; nothing after the chunk is simulated",
        "checkpoint_epoch": int(getattr(model, "checkpoint_epoch", -1)),
        "rollout_steps": 2 + HORIZON + HOLD, "own_hold": bool(args.own_hold),
        "classes_from": args.stage2, "probes": len(snippets), "scales": list(SCALES),
        "caveat": "the bundled checkpoint was trained with num_pred=1; on the toy system that "
                  "recipe hedged different outcomes toward the middle attractor"},
        "rows": rows}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({s: {"alarms": sum(r["by_scale"][s]["visual"]["alarm"] for r in rows)}
                      for s in rows[0]["by_scale"]}, indent=1))


if __name__ == "__main__":
    main()
