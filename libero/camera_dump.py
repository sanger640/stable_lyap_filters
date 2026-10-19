"""Agentview camera projection and robot base position per SafeLIBERO scenario (proprioception-level geometry for the
hand mask; no object state). Frames in L2 are flipped [::-1, ::-1], so pixel (u, v) -> (RES-1-u, RES-1-v).

usage (client env): python libero/camera_dump.py -> results/libero/camera.json
"""
import json
import pathlib

import numpy as np

RES = 392
SUITES = ("safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long")


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    out = {}
    for s in SUITES:
        for lv in ("I", "II"):
            suite = benchmark.get_benchmark_dict()[s](safety_level=lv)
            for tid in range(4):
                task = suite.get_task(tid)
                env = OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                                         camera_heights=64, camera_widths=64)
                env.reset(); env.set_init_state(suite.get_task_init_states(tid)[0])
                sim = env.sim
                M = get_camera_transform_matrix(sim, "agentview", RES, RES)
                base = sim.data.body_xpos[sim.model.body_name2id("robot0_base")].copy()
                out[f"{s}_{lv}_t{tid}"] = {"world_to_pixel": M.tolist(), "robot_base": base.tolist(), "flipped": True, "res": RES}
                env.close()
                print(s, lv, tid, flush=True)
    p = pathlib.Path("results/libero/camera.json"); p.write_text(json.dumps(out) + "\n"); print("done", p)


def project(entry, pts):
    """world points (N, 3) -> pixel (u = column, v = row) in the stored (flipped) frames, (N, 2). Checked visually on
    L2 frames: M gives (column, row) of the raw render; the [::-1, ::-1] flip of an upside-down render leaves rows and
    mirrors columns."""
    M = np.asarray(entry["world_to_pixel"]); P = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M.T
    c, r = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    return np.stack([entry["res"] - 1 - c, r], 1)


if __name__ == "__main__":
    main()
