"""PLAN_JENGA J1: load the bundled model and report autoregressive error growth."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, NUM_HIST, JengaReplay,
                           load_world_model, normalise_actions, normalise_proprio,
                           preprocess_frames)


def portable_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=str(ROOT / "results" / "jenga" / "j1_fidelity.json"))
    args = ap.parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or horizons[0] < 1:
        ap.error("horizons must be positive")

    replay = JengaReplay(args.lmdb)
    model = load_world_model(args.checkpoint, args.device)
    ids = replay.episode_ids[:min(args.episodes, len(replay.episode_ids))]
    errors = {h: [] for h in horizons}
    cosine = {h: [] for h in horizons}
    finite = True
    print(f"checkpoint epoch {model.checkpoint_epoch}; {len(ids)} episodes; device={args.device}")

    try:
        for ep_id in ids:
            ep = replay.episode(ep_id)
            need_frames = NUM_HIST + horizons[-1]
            if ep.length < need_frames:
                print(f"skip episode {ep_id}: only {ep.length} steps")
                continue
            frames = replay.frames(ep, range(need_frames))
            visual = preprocess_frames(frames, args.device)
            prop = normalise_proprio(ep.proprio[:need_frames], args.device).unsqueeze(0)
            # To predict H future frames from three observed frames, actions are required through
            # the step immediately before the Hth target frame.
            acts = normalise_actions(ep.actions[:NUM_HIST + horizons[-1] - 1],
                                     args.device).unsqueeze(0)
            with torch.inference_mode():
                truth = model.encode_obs({"visual": visual, "proprio": prop})["visual"]
                pred, _ = model.rollout(
                    {"visual": visual[:, :NUM_HIST], "proprio": prop[:, :NUM_HIST]}, acts)
                pred = pred["visual"]
            finite &= bool(torch.isfinite(pred).all())
            for h in horizons:
                target_index = NUM_HIST + h - 1
                delta = pred[:, target_index] - truth[:, target_index]
                scale = truth[:, target_index].float().std().clamp_min(1e-8)
                errors[h].append(float(torch.sqrt((delta.float() ** 2).mean()) / scale))
                c = torch.nn.functional.cosine_similarity(
                    pred[:, target_index].float(), truth[:, target_index].float(), dim=-1)
                cosine[h].append(float((1 - c).mean()))
            print(f"  episode {ep_id} done")
    finally:
        replay.close()

    summary = {}
    print("\nhorizon  normalised-RMSE  mean-cosine-distance")
    for h in horizons:
        summary[str(h)] = {
            "normalised_rmse": float(np.mean(errors[h])),
            "mean_cosine_distance": float(np.mean(cosine[h])),
        }
        print(f"{h:>7}  {summary[str(h)]['normalised_rmse']:>15.4f}  "
              f"{summary[str(h)]['mean_cosine_distance']:>20.5f}")
    print(f"\nfinite through horizon {horizons[-1]}: {finite}")
    if not finite:
        raise SystemExit("J1 FAIL: non-finite rollout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_epoch": model.checkpoint_epoch,
        "lmdb": portable_path(args.lmdb),
        "episodes": list(ids),
        "device": args.device,
        "finite": finite,
        "horizons": summary,
    }, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
