"""V1 gate: the frozen dynamics model driven by a state ESTIMATED from images.

Everything except the initial condition is unchanged from the privileged run -- same test states,
same 64 execution-noise probes, same scales, same score (spread of predicted endings), same
threshold rule (95th percentile of DEVELOPMENT quiet-state scores) and the same episode-clustered
intervals. The only substitution is

    step_state(sim)   ->   probe(DINO latents of the frames at the chunk start)

so a drop in recall is attributable to perception and state estimation, not to the dynamics, which
is loaded frozen from its checkpoint.

The probe, its PCA basis and the frames it reads are exactly those `jenga_v1_probe.py` fitted on
TRAINING configurations; this script renders and encodes the held-out scenes the same way
`jenga_bulk_data.py` did, so the encoder sees the same kind of input it was probed on.

Also reported, because it is the number that decides whether to build a visual world model at all:
how many fork states the estimated-state model is BLIND to -- predicted spread below the threshold
by a wide margin -- against the privileged model's count on the same states.
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
                           NUM_HIST, load_world_model)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import SCALES  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_v1_encode import preprocess as v1_preprocess  # noqa: E402
from jenga_v1_probe import GROUPS  # noqa: E402
from jenga_w5_gate3 import gate3, real_scores  # noqa: E402
from jenga_w5_eval import hold_index, load_model, predict  # noqa: E402


class Estimator:
    """DINO frames at the chunk start -> the 61-dim state, using the fitted probe."""

    def __init__(self, path, kind, device):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.mean = state["mean"].to(device)
        self.basis = state["basis"].to(device)
        self.frames = int(state["frames"])
        self.kind = kind
        self.device = device
        if kind == "linear":
            self.weights = state["linear"].to(device)
        else:
            width = state["mlp"]["0.weight"].shape[0]
            inputs = state["mlp"]["0.weight"].shape[1]
            outputs = state["mlp"]["4.weight"].shape[0]
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(inputs, width), torch.nn.GELU(),
                torch.nn.Linear(width, width), torch.nn.GELU(),
                torch.nn.Linear(width, outputs)).to(device)
            self.mlp.load_state_dict(state["mlp"])
            self.mlp.eval()

    def __call__(self, latents):
        """latents (frames, 196*384) float32 -> (61,) numpy."""
        x = torch.as_tensor(latents, device=self.device, dtype=torch.float32)
        features = ((x - self.mean) @ self.basis).reshape(1, -1)
        with torch.no_grad():
            if self.kind == "linear":
                out = torch.cat([features, torch.ones(1, 1, device=self.device)], 1) @ self.weights
            else:
                out = self.mlp(features)
        return out[0].cpu().numpy()


def encode_frames(encoder, device, frames, pixels):
    """Direct DINOv2 call at `pixels`, matching jenga_v1_encode -- no model-side 196 px resize."""
    with torch.inference_mode():
        z = encoder.forward(v1_preprocess(np.stack(frames), device, pixels))
    return z.float().flatten(1).cpu().numpy()


def model_scores(rows, dynamics, scale, snippets, lmdb, sim_archive, device, reset_base,
                 estimator=None, encoder=None, collect=None, true_groups=(), pixels=196,
                 render=(240, 320)):
    """Predicted ending spread per state and error size; estimated state when a probe is given."""
    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r
    replay = JengaReplay(lmdb)
    out = []
    with tempfile.TemporaryDirectory(prefix="jenga_v1_") as temp:
        sim = DirectJengaSim(str(extract_sim(sim_archive, temp)))
        if tuple(render) != (240, 320):
            import mujoco
            sim.renderer.close()
            sim.renderer = mujoco.Renderer(sim.model, height=render[0], width=render[1])
        try:
            for episode_id in sorted(wanted, key=int):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id) + reset_base if reset_base is not None
                          else int(episode_id))
                history = [sim.render()]
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        truth = step_state(sim)
                        if estimator is None:
                            start = truth
                        else:
                            window = history[-estimator.frames:]
                            while len(window) < estimator.frames:
                                window = [window[0]] + window
                            latents = encode_frames(encoder, device, window, pixels)
                            start = estimator(latents)
                            # Ablation: hand back the TRUE value of some groups, to find which
                            # estimation error the monitor is actually sensitive to.
                            for name in true_groups:
                                start[GROUPS[name][0]] = truth[GROUPS[name][0]]
                            if collect is not None:
                                collect.append({"episode_id": episode_id, "chunk_start": step,
                                                "position_error_mm": float(np.sqrt(np.mean(
                                                    (1000 * (start[:9] - truth[:9])) ** 2)))})
                        for s in SCALES:
                            windows = action_windows(episode.actions, step, snippets, s,
                                                     own_hold=False)
                            ends = 1000 * predict(dynamics, scale, start, windows,
                                                  device)[:, hold_index(dynamics, 30), 0:9]
                            spread = float(np.sqrt(np.mean(np.sum(
                                (ends - ends.mean(0)) ** 2, 1))))
                            out.append({"episode_id": episode_id, "chunk_start": step,
                                        "scale": str(s), "score": spread,
                                        "class": row["by_scale"][str(s)]["class"]})
                    sim.execute(action)
                    history.append(sim.render())
                    history = history[-max(NUM_HIST, estimator.frames if estimator else 1):]
        finally:
            sim.close()
            replay.close()
    return out


def blind_forks(scores, threshold, margin=0.5):
    """Fork states the model is not merely missing but BLIND to: score below margin x threshold."""
    forks = [r for r in scores if r["class"] == "topple_fork"]
    if not forks:
        return None
    blind = [r for r in forks if r["score"] < margin * threshold]
    return {"fork_states": len(forks), "blind": len(blind),
            "rate": len(blind) / len(forks),
            "median_score_of_blind": float(np.median([r["score"] for r in blind]))
            if blind else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="frozen dynamics checkpoint")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--probe-kind", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--dev", default=str(ROOT / "results/jenga/holdout_stage2_shared.json"))
    ap.add_argument("--test", default=str(ROOT / "results/jenga/holdout3_stage2_shared.json"))
    ap.add_argument("--test-reset-seed-base", type=int, default=1000)
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--pixels", type=int, default=196)
    ap.add_argument("--render", type=int, nargs=2, default=(240, 320),
                    metavar=("HEIGHT", "WIDTH"))
    ap.add_argument("--true-groups", nargs="*", default=["gripper"], choices=list(GROUPS),
                    help="default is gripper: end-effector pose is proprioception at deployment, "
                         "not something vision should be asked to reconstruct",
                    help="state groups to take from the simulator instead of the probe")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dynamics, scale = load_model(args.model, device)
    encoder = load_world_model(args.checkpoint, device).encoder
    estimator = Estimator(args.probe, args.probe_kind, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    dev_rows = json.loads(Path(args.dev).read_text())["rows"]
    test_rows = json.loads(Path(args.test).read_text())["rows"]

    accuracy = []
    dev = model_scores(dev_rows, dynamics, scale, snippets, args.lmdb, args.sim_archive, device,
                       None, estimator, encoder, true_groups=args.true_groups,
                       pixels=args.pixels, render=args.render)
    test = model_scores(test_rows, dynamics, scale, snippets, args.lmdb, args.sim_archive, device,
                        args.test_reset_seed_base, estimator, encoder, accuracy,
                        true_groups=args.true_groups, pixels=args.pixels, render=args.render)
    result = {"protocol": {"pipeline": "DINO latents -> probe -> frozen dynamics",
                           "probe": args.probe, "probe_kind": args.probe_kind,
                           "dynamics": args.model,
                           "unchanged": "test states, probes, scales, score, threshold rule, "
                                        "intervals"},
              "true_groups": args.true_groups,
              "state_estimation": {
                  "position_rmse_mm_median": float(np.median(
                      [a["position_error_mm"] for a in accuracy])) if accuracy else None},
              "model": gate3(dev, test),
              "real_endings_reference": gate3(real_scores(dev_rows), real_scores(test_rows))}
    for s in map(str, SCALES):
        result["model"][s]["blind_forks"] = blind_forks(
            [r for r in test if r["scale"] == s], result["model"][s]["threshold"])
    result["scores"] = test
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}, indent=1)[:1600])


if __name__ == "__main__":
    main()
