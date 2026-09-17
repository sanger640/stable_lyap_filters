"""Chunk-local Jenga experiment: H=8 followed by held tails of 0, 5, or 10 steps.

Recorded absolute-EE actions are replayed in the bundled MuJoCo scene. Every eight-step chunk is
branched at its endpoint: keep the last pose/gripper command for 0, 5, or 10 control periods, render
the true endpoint, and roll the same branch through the world model. The branch is then discarded
and nominal replay resumes, so later policy actions never leak into the chunk's measured effect.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from basins import ProprioResidualizer, fit_basin_model             # noqa: E402
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay, load_world_model,
                           normalise_actions, normalise_proprio, preprocess_frames)  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_j2_j3_predicted_basins import best_agreement, label_stability, separation_ratio  # noqa: E402
from jenga_j4_hedging import axis_cosine, outcome_axis              # noqa: E402

HORIZON = 8
TAILS = (0, 5, 10)
TOPPLE_DEG = 45.0
BLOCK_NAMES = ("block_middle", "block_left", "block_right")
CONTACT_LABELS = (
    "middle-left", "middle-right", "left-right",
    "middle-table", "left-table", "right-table",
    "middle-floor", "left-floor", "right-floor",
    "middle-robot", "left-robot", "right-robot",
)


def chunk_starts(n_actions, horizon=HORIZON):
    """Non-overlapping future chunks with frames 0/1/2 as the first history."""
    return list(range(2, int(n_actions) - horizon + 1, horizon))


def model_action_window(actions, start, tail):
    """Two historical actions + H future actions + repeated final-pose tail."""
    a = np.asarray(actions, np.float32)
    if start < 2 or start + HORIZON > len(a):
        raise ValueError("chunk does not have two history actions and eight future actions")
    chunk = a[start:start + HORIZON]
    held = np.repeat(chunk[-1:], int(tail), axis=0)
    return np.concatenate([a[start - 2:start], chunk, held])


class DirectJengaSim:
    """Fast deterministic version of panda_express/sim.py without its real-time thread."""

    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("attachment_site").id
        self.tracked_block_ids = [self.model.body(x).id for x in BLOCK_NAMES]
        self.block_ids = self.tracked_block_ids[1:]
        self.table_id = self.model.body("table").id
        self.steps_per_action = int(round(0.1 / self.model.opt.timestep))
        self.orientation = np.zeros(4)
        self.target = np.zeros(3)
        self.gripper = 110.0
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.segment_renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.segment_renderer.enable_segmentation_rendering()
        self.block_geom_ids = np.flatnonzero(np.isin(
            self.model.geom_bodyid, self.tracked_block_ids))
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.model, self.state_spec)

    def close(self):
        self.renderer.close()
        self.segment_renderer.close()

    def reset(self, seed):
        m, d, mj = self.model, self.data, self.mj
        mj.mj_resetDataKeyframe(m, d, 0)
        rng = np.random.default_rng(int(seed))
        for name in ("block_middle", "block_left", "block_right"):
            body = m.body(name).id
            joint = m.body_jntadr[body]
            adr = m.jnt_qposadr[joint]
            d.qpos[adr:adr + 2] += rng.uniform(-0.002, 0.002, 2)
            yaw = np.radians(rng.uniform(-3.0, 3.0))
            half = yaw / 2
            noise = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            old = d.qpos[adr + 3:adr + 7].copy()
            new = np.zeros(4)
            mj.mju_mulQuat(new, noise, old)
            d.qpos[adr + 3:adr + 7] = new
        mj.mj_forward(m, d)
        self.target = d.site_xpos[self.site].copy()
        mj.mju_mat2Quat(self.orientation, d.site_xmat[self.site])
        self.gripper = 110.0

    def snapshot(self):
        state = np.empty(self.state_size)
        self.mj.mj_getState(self.model, self.data, state, self.state_spec)
        return state, self.data.ctrl.copy(), self.target.copy(), float(self.gripper)

    def restore(self, snapshot):
        state, ctrl, target, gripper = snapshot
        self.mj.mj_setState(self.model, self.data, state, self.state_spec)
        self.data.ctrl[:] = ctrl
        self.target = target.copy()
        self.gripper = gripper
        self.mj.mj_forward(self.model, self.data)

    def _control(self):
        m, d, mj = self.model, self.data, self.mj
        current_matrix = d.site_xmat[self.site].reshape(3, 3)
        target_matrix = np.zeros(9)
        mj.mju_quat2Mat(target_matrix, self.orientation)
        error = target_matrix.reshape(3, 3) @ current_matrix.T
        dr = np.array([error[2, 1] - error[1, 2], error[0, 2] - error[2, 0],
                       error[1, 0] - error[0, 1]]) * 0.5
        jac = np.zeros((6, m.nv))
        mj.mj_jacSite(m, d, jac[:3], jac[3:], self.site)
        velocity = jac @ d.qvel
        wrench = np.hstack([500 * (self.target - d.site_xpos[self.site]) - 30 * velocity[:3],
                            20 * dr - 0.5 * velocity[3:]])
        d.ctrl[:7] = (jac.T @ wrench - 0.1 * d.qvel)[:7]
        d.ctrl[7] = self.gripper

    def block_diagnostics(self):
        rotations = self.data.xmat[self.tracked_block_ids].reshape(-1, 3, 3)
        tilts = np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))
        return {"position": self.data.xpos[self.tracked_block_ids].copy(),
                "quaternion": self.data.xquat[self.tracked_block_ids].copy(),
                "tilt": tilts.astype(np.float64)}

    def contact_signature(self):
        """Block/block, support, and block/robot contact categories."""
        signature = np.zeros(len(CONTACT_LABELS), bool)
        block_index = {body: i for i, body in enumerate(self.tracked_block_ids)}
        for contact in self.data.contact[:self.data.ncon]:
            bodies = (int(self.model.geom_bodyid[contact.geom1]),
                      int(self.model.geom_bodyid[contact.geom2]))
            blocks = [block_index[body] for body in bodies if body in block_index]
            if len(blocks) == 2:
                pair = tuple(sorted(blocks))
                signature[{(0, 1): 0, (0, 2): 1, (1, 2): 2}[pair]] = True
            elif len(blocks) == 1:
                block = blocks[0]; other = bodies[0] if bodies[1] in block_index else bodies[1]
                if other == self.table_id:
                    signature[3 + block] = True
                elif other == 0:
                    signature[6 + block] = True
                else:
                    signature[9 + block] = True
        return signature

    def _observe_trace(self, trace):
        blocks = self.block_diagnostics()
        trace["peak_tilt"] = np.maximum(trace["peak_tilt"], blocks["tilt"])
        if "min_z" in trace:
            trace["min_z"] = np.minimum(trace["min_z"], blocks["position"][:, 2])
        if "max_z" in trace:
            trace["max_z"] = np.maximum(trace["max_z"], blocks["position"][:, 2])
        if "contact_seen" in trace:
            contact = self.contact_signature()
            trace["contact_seen"] |= contact
            trace["contact_transitions"] += contact != trace["previous_contact"]
            trace["previous_contact"] = contact

    def execute(self, action, trace=None):
        action = np.asarray(action, np.float32)
        self.target = action[:3].astype(float)
        if action[3] > 0.9:
            self.gripper = 0.0
        elif action[3] < -0.9:
            self.gripper = 110.0
        peak = self.max_tilt()
        if trace is not None:
            self._observe_trace(trace)
        for _ in range(self.steps_per_action):
            self._control()
            self.mj.mj_step(self.model, self.data)
            peak = max(peak, self.max_tilt())
            if trace is not None:
                self._observe_trace(trace)
        return peak

    def max_tilt(self):
        values = []
        for body in self.block_ids:
            rotation = self.data.xmat[body].reshape(3, 3)
            values.append(float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))))
        return max(values)

    def proprio(self):
        closed = 1.0 if self.gripper < 50 else -1.0
        return np.r_[self.data.site_xpos[self.site], closed].astype(np.float32)

    def render(self):
        self.renderer.update_scene(self.data, camera="cam_fixed")
        return self.renderer.render().copy()

    def block_patch_mask(self, dilation=1):
        """14x14 DINO patch mask from MuJoCo block geometry segmentation."""
        self.segment_renderer.update_scene(self.data, camera="cam_fixed")
        segmentation = self.segment_renderer.render()
        pixels = ((segmentation[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM))
                  & np.isin(segmentation[..., 0], self.block_geom_ids))
        mask = np.zeros((14, 14), bool)
        ys, xs = np.nonzero(pixels)
        if len(ys):
            mask[np.minimum(ys * 14 // pixels.shape[0], 13),
                 np.minimum(xs * 14 // pixels.shape[1], 13)] = True
        for _ in range(int(dilation)):
            expanded = mask.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    expanded |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            mask = expanded
        return mask


def extract_sim(archive, destination):
    with tarfile.open(archive) as bundle:
        root = Path(destination).resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe simulator archive member {member.name}")
        bundle.extractall(destination)
    return Path(destination) / "panda_express" / "franka_emika_panda" / "panda_jenga_setup.xml"


def projection_basis(path, dimension=16):
    """Label-free PCA basis from the 100 real terminal frames already encoded by J4."""
    cache = dict(np.load(path, allow_pickle=False))
    endings = np.asarray(cache["tail_truth"][:, -1], np.float32).reshape(
        len(cache["tail_truth"]), -1)
    mean = endings.mean(0)
    components = np.linalg.svd(endings - mean, full_matrices=False)[2][:dimension]
    return mean.astype(np.float32), components.astype(np.float32)


def predict_episode(model, device, records, projection):
    if not records:
        return {tail: ([], []) for tail in TAILS}
    frames = np.stack([r["history_frames"] for r in records])
    props = np.stack([r["history_proprio"] for r in records])
    initial = {"visual": preprocess_frames(frames, device),
               "proprio": normalise_proprio(props, device)}
    mean, components = projection
    mean = torch.as_tensor(mean, device=device)
    components = torch.as_tensor(components, device=device)
    actions = np.stack([r["action_windows"][TAILS[-1]] for r in records])
    true_frames = np.stack([
        r["tail_frames"][tail] for r in records for tail in TAILS])[:, None]
    true_props = np.stack([
        r["tail_proprio"][tail] for r in records for tail in TAILS])[:, None]
    with torch.inference_mode():
        prediction, _ = model.rollout(initial, normalise_actions(actions, device))
        actual = model.encode_obs({"visual": preprocess_frames(true_frames, device),
                                   "proprio": normalise_proprio(true_props, device)})["visual"]
    actual = actual[:, 0].float().reshape(len(records), len(TAILS), -1)
    output = {}
    for tail_index, tail in enumerate(TAILS):
        # Three observed frames plus H+tail transitions ends at output index 10+tail.
        pred_flat = prediction["visual"][:, 10 + tail].float().flatten(1)
        real_flat = actual[:, tail_index]
        output[tail] = (((pred_flat - mean) @ components.T).cpu().numpy(),
                        ((real_flat - mean) @ components.T).cpu().numpy())
    return output


def collect(args, replay, ids, model, xml, projection):
    sim = DirectJengaSim(xml)
    all_data = {tail: {k: [] for k in ("predicted", "actual", "proprio", "end_tilt",
                                       "peak_tilt", "new_topple")} for tail in TAILS}
    episodes, starts = [], []
    pending = []

    def flush(records):
        predicted = predict_episode(model, args.device, records, projection)
        for record_index, record in enumerate(records):
            for tail in TAILS:
                pred, actual = predicted[tail]
                values = all_data[tail]
                values["predicted"].append(pred[record_index].reshape(-1))
                values["actual"].append(actual[record_index].reshape(-1))
                values["proprio"].append(record["tail_proprio"][tail])
                for key in ("end_tilt", "peak_tilt", "new_topple"):
                    values[key].append(record[key][tail])

    try:
        for number, ep_id in enumerate(ids):
            episode = replay.episode(ep_id)
            actions = episode.actions
            sim.reset(int(ep_id))
            frames = deque([sim.render()], maxlen=3)
            props = deque([sim.proprio()], maxlen=3)
            # Establish frames 1 and 2. Action i maps frame i to frame i+1.
            for i in range(2):
                sim.execute(actions[i]); frames.append(sim.render()); props.append(sim.proprio())
            records = []
            for start in chunk_starts(len(actions)):
                if start > 2:
                    # The previous chunk ended at frame=start; histories are already current.
                    assert len(frames) == 3
                record = {"history_frames": np.stack(frames),
                          "history_proprio": np.stack(props), "action_windows": {},
                          "tail_frames": {}, "tail_proprio": {}, "end_tilt": {},
                          "peak_tilt": {}, "new_topple": {}}
                start_tilt = sim.max_tilt(); peak = start_tilt
                for action_index, action in enumerate(actions[start:start + HORIZON]):
                    peak = max(peak, sim.execute(action))
                    if action_index >= HORIZON - 3:
                        frames.append(sim.render()); props.append(sim.proprio())
                branch = sim.snapshot()
                for tail in TAILS:
                    sim.restore(branch); branch_peak = peak
                    for _ in range(tail):
                        branch_peak = max(branch_peak, sim.execute(actions[start + HORIZON - 1]))
                    end_tilt = sim.max_tilt()
                    record["action_windows"][tail] = model_action_window(actions, start, tail)
                    record["tail_frames"][tail] = sim.render()
                    record["tail_proprio"][tail] = sim.proprio()
                    record["end_tilt"][tail] = end_tilt
                    record["peak_tilt"][tail] = branch_peak
                    record["new_topple"][tail] = bool(start_tilt < TOPPLE_DEG <= branch_peak)
                sim.restore(branch)
                records.append(record); episodes.append(ep_id); starts.append(start)
            pending.extend(records)
            while len(pending) >= args.batch_size:
                flush(pending[:args.batch_size]); del pending[:args.batch_size]
            print(f"  {number + 1}/{len(ids)} ep{ep_id}: {len(records)} chunks", flush=True)
        if pending:
            flush(pending)
    finally:
        sim.close()
    packed = {"episode_ids": np.asarray(episodes), "chunk_starts": np.asarray(starts, np.int32)}
    for tail, values in all_data.items():
        for key, value in values.items():
            dtype = np.float16 if key in ("predicted", "actual") else None
            packed[f"tail{tail}_{key}"] = np.asarray(value, dtype=dtype)
    return packed


def low_pca(points, dimension=2):
    from sklearn.decomposition import PCA
    return PCA(n_components=dimension, svd_solver="randomized", random_state=0).fit_transform(
        np.asarray(points, np.float32))


def geometry(points, proprio, toppled, tilt, min_fraction):
    residualizer = ProprioResidualizer.fit(points, proprio)
    residual = residualizer.transform(points, proprio)
    x = low_pca(residual, 2)
    correlation = float(np.corrcoef(
        np.linalg.norm(x - x[toppled == 0].mean(0), axis=1), tilt)[0, 1])
    summary = {"separation_ratio": separation_ratio(x, toppled),
               "tilt_correlation_pc_distance": correlation if np.isfinite(correlation) else None}
    try:
        basin = fit_basin_model(x, 2, min_fraction)
        agreement, coverage = best_agreement(basin.labels, toppled)
        summary.update({"n_clusters": basin.n_clusters,
                        "agreement": agreement if np.isfinite(agreement) else None,
                        "coverage": coverage,
                        "cluster_sizes": sorted(basin.cluster_sizes, reverse=True)})
    except ValueError as ex:
        basin = None
        summary.update({"n_clusters": 0, "agreement": 0.0, "coverage": 0.0,
                        "cluster_sizes": [], "error": str(ex)})
    return residual, basin, summary


def terminal_reference(projection_cache, j2_cache, labels_path, projection, min_fraction=0.10):
    encoded = dict(np.load(projection_cache, allow_pickle=False))
    j2 = dict(np.load(j2_cache, allow_pickle=False))
    ids = [str(x) for x in encoded["episode_ids"]]
    labels = json.load(open(labels_path))
    truth = np.array([labels[e]["outcome"] != "success" for e in ids], int)
    endings = np.asarray(encoded["tail_truth"][:, -1], np.float32).reshape(len(ids), -1)
    mean, components = projection
    projected = (endings - mean) @ components.T
    proprio = np.asarray(j2["proprio"][:, -1], np.float32)
    residualizer = ProprioResidualizer.fit(projected, proprio)
    residual = residualizer.transform(projected, proprio)
    basin = fit_basin_model(residual, 2, min_fraction)
    agreement, coverage = best_agreement(basin.labels, truth)
    return residualizer, basin, {
        "episodes": len(ids), "n_clusters": basin.n_clusters, "agreement": agreement,
        "coverage": coverage, "cluster_sizes": sorted(basin.cluster_sizes, reverse=True),
        "min_cluster_fraction": min_fraction,
    }


def assign_reference(points, proprio, toppled, residualizer, basin):
    residual = residualizer.transform(points, proprio)
    labels = basin.predict(residual)
    agreement, coverage = best_agreement(labels, toppled)
    summary = {"agreement": agreement if np.isfinite(agreement) else None,
               "coverage": coverage,
               "cluster_counts": [int((labels == c).sum()) for c in range(basin.n_clusters)],
               "noise": int((labels < 0).sum()), "by_physical_outcome": {}}
    for value, name in ((0, "intact"), (1, "toppled")):
        member = toppled == value
        summary["by_physical_outcome"][name] = {
            "n": int(member.sum()), "coverage": float((labels[member] >= 0).mean()),
            "cluster_counts": [int(((labels == c) & member).sum())
                               for c in range(basin.n_clusters)],
        }
    return labels, summary


def analyse(data, min_fraction, reference):
    reference_residualizer, reference_basin, reference_summary = reference
    result = {"chunks": len(data["episode_ids"]), "episodes": len(set(data["episode_ids"].astype(str))),
              "horizon": HORIZON, "terminal_reference": reference_summary, "tails": {}}
    reference_labels = {}
    # Consecutive non-overlapping chunks: the prior tail-0 endpoint is the next start state.
    start_tilt = np.zeros(len(data["episode_ids"]), np.float32)
    previous_episode = None
    previous_tilt = 0.0
    for i, episode in enumerate(data["episode_ids"].astype(str)):
        if episode != previous_episode:
            previous_tilt = 0.0
        start_tilt[i] = previous_tilt
        previous_tilt = float(data["tail0_end_tilt"][i])
        previous_episode = episode
    eligible = start_tilt < TOPPLE_DEG
    for tail in TAILS:
        pred = np.asarray(data[f"tail{tail}_predicted"], np.float32)
        actual = np.asarray(data[f"tail{tail}_actual"], np.float32)
        prop = data[f"tail{tail}_proprio"]
        tilt = np.asarray(data[f"tail{tail}_end_tilt"], np.float32)
        peak = np.asarray(data[f"tail{tail}_peak_tilt"], np.float32)
        toppled = (peak >= TOPPLE_DEG).astype(int)
        real_res, real_basin, real = geometry(actual, prop, toppled, tilt, min_fraction)
        pred_res, pred_basin, predicted = geometry(pred, prop, toppled, tilt, min_fraction)
        real_axis, real_direction = outcome_axis(real_res, toppled)
        pred_axis, pred_direction = outcome_axis(pred_res, toppled)
        actual_reference_labels, actual_reference = assign_reference(
            actual, prop, toppled, reference_residualizer, reference_basin)
        predicted_reference_labels, predicted_reference = assign_reference(
            pred, prop, toppled, reference_residualizer, reference_basin)
        reference_labels[tail] = {"actual": actual_reference_labels,
                                  "predicted": predicted_reference_labels}
        error = float(np.sqrt(np.mean((pred - actual) ** 2)))
        result["tails"][str(tail)] = {
            "toppled_endpoints": int(toppled.sum()),
            "new_topples": int(np.asarray(data[f"tail{tail}_new_topple"]).sum()),
            "eligible_upright_chunk_starts": int(eligible.sum()),
            "new_topple_rate": float(np.asarray(data[f"tail{tail}_new_topple"])[eligible].mean()),
            "mean_end_tilt_deg": float(tilt.mean()), "max_end_tilt_deg": float(tilt.max()),
            "prediction_rmse": error,
            "prediction_normalised_rmse": float(error / max(float(actual.std()), 1e-12)),
            "outcome_axis_contraction": float(pred_axis / max(real_axis, 1e-12)),
            "outcome_axis_cosine": axis_cosine(real_direction, pred_direction),
            "real": real, "predicted": predicted,
            "terminal_basin_assignment": {"actual": actual_reference,
                                           "predicted": predicted_reference},
        }
    for tail in TAILS[1:]:
        for source in ("actual", "predicted"):
            score, coverage = label_stability(
                reference_labels[0][source], reference_labels[tail][source])
            result["tails"][str(tail)][f"{source}_terminal_assignment_stability_from_tail0"] = {
                "agreement": score, "joint_coverage": coverage}
    return result


def replay_validation(data, labels_path):
    """How closely re-sampled simulator resets reproduce the original episode outcomes."""
    labels = json.load(open(labels_path))
    ids = data["episode_ids"].astype(str)
    simulated, original = [], []
    for episode in sorted(set(ids), key=int):
        simulated.append(bool(np.any(data["tail0_peak_tilt"][ids == episode] >= TOPPLE_DEG)))
        original.append(labels[episode]["outcome"] != "success")
    simulated = np.asarray(simulated, bool); original = np.asarray(original, bool)
    return {"simulated_toppled": int(simulated.sum()), "original_toppled": int(original.sum()),
            "agreement": float((simulated == original).mean()),
            "confusion": {"tp": int((simulated & original).sum()),
                          "fp": int((simulated & ~original).sum()),
                          "fn": int((~simulated & original).sum()),
                          "tn": int((~simulated & ~original).sum())},
            "note": "actions are replayed under deterministic re-sampled reset variation; original MuJoCo states were not recorded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--projection-cache", default=str(
        ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--j2-cache", default=str(
        ROOT / "results/jenga/j2_j3_predicted_endings.npz"))
    ap.add_argument("--labels", default=str(ROOT / "data/jenga/labels_noise100.json"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-cluster-fraction", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/short_held_tails_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/short_held_tails.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    cache = Path(args.cache)
    projection = projection_basis(args.projection_cache)
    reference = terminal_reference(args.projection_cache, args.j2_cache, args.labels,
                                   projection)
    if args.reuse_cache and cache.exists():
        data = dict(np.load(cache, allow_pickle=False)); print(f"loaded {cache}")
    else:
        replay = JengaReplay(args.lmdb)
        ids = replay.episode_ids[:min(args.episodes, len(replay.episode_ids))]
        model = load_world_model(args.checkpoint, args.device)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_sim_") as temp:
                xml = extract_sim(args.sim_archive, temp)
                data = collect(args, replay, ids, model, xml, projection)
        finally:
            replay.close()
        cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, **data)
        print(f"cached -> {cache}")
    result = analyse(data, args.min_cluster_fraction, reference)
    result.update({"min_cluster_fraction": args.min_cluster_fraction,
                   "topple_threshold_deg": TOPPLE_DEG,
                   "definition": "8 recorded actions, then repeat final absolute pose/gripper command",
                   "latent_projection": "label-free PCA16 fitted to J4's 100 encoded real final frames",
                   "replay_validation": replay_validation(data, args.labels)})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("\nheld-tail result")
    print("tail chunks toppled new  intrinsic real/pred sep  terminal coverage real/pred  RMSE  axis-scale/cos")
    for tail in TAILS:
        row = result["tails"][str(tail)]; real = row["real"]; pred = row["predicted"]
        assignment = row["terminal_basin_assignment"]
        print(f"{tail:>4}{result['chunks']:>7}{row['toppled_endpoints']:>8}{row['new_topples']:>4}  "
              f"{real['separation_ratio']:.2f}/{pred['separation_ratio']:.2f}"
              f"                 {assignment['actual']['coverage']:.1%}/"
              f"{assignment['predicted']['coverage']:.1%}"
              f"   {row['prediction_rmse']:.3f}  {row['outcome_axis_contraction']:.2f}/"
              f"{row['outcome_axis_cosine']:.2f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
