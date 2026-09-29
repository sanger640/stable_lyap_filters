"""Program B gate G0: does the FIGNet* port beat a plain object-level GNS? (addendum 2)

400 twenty-step windows (starts 2, 22, 42, 62) from the 100 generic TEST trajectories. Primary:
mean object position error over windows, steps and free objects. Interpenetration: per-step
maximum penetration depth over object/object and object/table pairs (coal convex signed
distance), averaged over windows and steps. Pass: median FIGNet* (3 seeds) position error below
GNS seed 1, and median FIGNet* EXCESS interpenetration (predicted minus the true trajectory on the
same windows; addendum 3) below 1 mm. Final checkpoints only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import coal
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import program_b_fignet as pf  # noqa: E402
import program_b_gns as pg  # noqa: E402
from program_b_train import CHECKPOINTS, load  # noqa: E402

CORPUS = ROOT / "results/program_b/corpus"
ADDENDUM = ROOT / "results/jenga/program_b_stage1_addendum_2.json"
ADDENDUM_3 = ROOT / "results/jenga/program_b_stage1_addendum_3.json"
OUTPUT = ROOT / "results/jenga/program_b_g0_result.json"
STARTS = (2, 22, 42, 62)
HORIZON = 20
FINAL_STEP = 400_000
RUNS = [("fignet", 1), ("fignet", 2), ("fignet", 3), ("gns", 1)]


def convex(mesh):
    vertices = coal.StdVec_Vec3s(); vertices.extend(list(np.asarray(mesh.vertices, float)))
    triangles = coal.StdVec_Triangle()
    triangles.extend([coal.Triangle(int(a), int(b), int(c)) for a, b, c in mesh.faces])
    return coal.Convex(vertices, triangles)


class Penetration:
    """Maximum penetration (m) among object/object and object/table pairs of one configuration."""

    def __init__(self, objects):
        meshes = [pf.object_mesh(o) for o in objects]
        self.shapes = [convex(m) for m in meshes]
        self.radii = [float(np.linalg.norm(m.vertices, axis=1).max()) for m in meshes]
        self.table = convex(trimesh.creation.box(extents=[0.4, 0.8, 0.05]))
        self.table_pose = coal.Transform3s(np.eye(3), np.array(pf.TABLE_POSE[:3]))

    @staticmethod
    def transform(pose_xyzw):
        return coal.Transform3s(Rotation.from_quat(pose_xyzw[3:]).as_matrix(),
                                np.asarray(pose_xyzw[:3], float))

    def __call__(self, poses_xyzw):
        n = len(self.shapes)
        transforms = [self.transform(p) for p in poses_xyzw[:n]]
        worst = 0.0
        request = coal.DistanceRequest()
        for i in range(n):
            pairs = [(self.table, self.table_pose, np.inf)]
            pairs += [(self.shapes[j], transforms[j], self.radii[i] + self.radii[j])
                      for j in range(i + 1, n)]
            for j, (shape, pose, reach) in enumerate(pairs):
                if j and np.linalg.norm(poses_xyzw[i, :3] - poses_xyzw[i + j, :3]) > reach:
                    continue
                distance = coal.distance(self.shapes[i], transforms[i], shape, pose, request,
                                         coal.DistanceResult())
                worst = max(worst, -distance)
        return worst


def rotation_error(predicted, truth):
    return (Rotation.from_quat(predicted.reshape(-1, 4)).inv()
            * Rotation.from_quat(truth.reshape(-1, 4))).magnitude().reshape(predicted.shape[:-1])


def evaluate(model, checkpoint, files, device):
    net = load(model, checkpoint, device)
    rows, step_seconds = [], []
    for path in files:
        meta, sequences = pf.episode_sequences(path)
        objects = meta["scene"]["objects"]
        n = len(objects)
        poses, actions, closed = sequences[0]
        penetration = Penetration(objects)
        for start in STARTS:
            began = time.time()
            if model == "fignet":
                scene, obj_ids = pf.build_scene(objects, noise_std=0.0)
                predicted = pf.rollout(net, scene, obj_ids, poses, actions, closed, start, HORIZON, device)
            else:
                predicted = pg.rollout(net, objects, poses, actions, closed, start, HORIZON, device)
            step_seconds.append((time.time() - began) / HORIZON)
            truth = poses[start + 1:start + 1 + HORIZON]
            position = np.linalg.norm(predicted[:, :n, :3] - truth[:, :n, :3], axis=-1)
            truth_pen = float(np.mean([penetration(p) for p in truth]))
            predicted_pen = float(np.mean([penetration(p) for p in predicted]))
            rows.append({
                "episode": meta["episode"], "start": start,
                "position_error": float(position.mean()),
                "position_error_step20": float(position[-1].mean()),
                "rotation_error": float(rotation_error(predicted[:, :n, 3:], truth[:, :n, 3:]).mean()),
                "hand_error": float(np.linalg.norm(predicted[:, n, :3] - truth[:, n, :3], axis=-1).mean()),
                "interpenetration": predicted_pen,
                "truth_interpenetration": truth_pen,
                "excess_interpenetration": predicted_pen - truth_pen,
            })
    summary = {key: float(np.mean([r[key] for r in rows]))
               for key in ("position_error", "position_error_step20", "rotation_error",
                           "hand_error", "interpenetration", "truth_interpenetration",
                           "excess_interpenetration")}
    summary["ms_per_step"] = 1000 * float(np.mean(step_seconds))
    summary["ms_per_64x38_batch"] = summary["ms_per_step"] * 64 * 38
    summary["windows"] = len(rows)
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="validation is for smoke tests only; G0 is measured on test")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--step", type=int, default=FINAL_STEP)
    parser.add_argument("--runs", nargs="*", default=None, help="e.g. gns:1 fignet:2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    files = sorted((CORPUS / args.split).glob("ep*.npz"))[:args.episodes]
    runs = [(r.split(":")[0], int(r.split(":")[1])) for r in args.runs] if args.runs else RUNS
    results = {}
    for model, seed in runs:
        checkpoint = CHECKPOINTS / f"{model}_s{seed}" / f"step_{args.step:07d}.pt"
        summary, rows = evaluate(model, checkpoint, files, args.device)
        results[f"{model}_s{seed}"] = {"checkpoint": str(checkpoint.relative_to(ROOT)),
                                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                       "summary": summary, "windows": rows}
        print(model, seed, json.dumps(summary), flush=True)
    fignet = [v["summary"] for k, v in results.items() if k.startswith("fignet")]
    gns = results.get("gns_s1", {}).get("summary")
    decision = None
    if len(fignet) == 3 and gns:
        median_error = float(np.median([s["position_error"] for s in fignet]))
        median_excess = float(np.median([s["excess_interpenetration"] for s in fignet]))
        decision = {"fignet_median_position_error": median_error,
                    "gns_position_error": gns["position_error"],
                    "fignet_median_excess_interpenetration_mm": 1000 * median_excess,
                    "truth_interpenetration_mm": 1000 * fignet[0]["truth_interpenetration"],
                    "passes": bool(median_error < gns["position_error"] and median_excess < 0.001)}
    result = {"addendum_sha256": hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(),
              "addendum_3_sha256": hashlib.sha256(ADDENDUM_3.read_bytes()).hexdigest(),
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "split": args.split, "episodes": len(files), "starts": STARTS, "horizon": HORIZON,
              "runs": results, "decision": decision}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": decision}, indent=2))


if __name__ == "__main__":
    main()
