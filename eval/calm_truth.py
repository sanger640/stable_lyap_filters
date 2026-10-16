"""Real risk (32 futures, cached draws) of the policy's own chunk at the 150 calm (untouched) dev3A decisions in the
cbf_qp cache, so trigger quality can be measured over all 361 cached decisions."""
import json, os, sys, multiprocessing as mp
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
import cbf_qp as q
if __name__ == "__main__":
    C, K = q.load_cache(); os.environ["TRUTH_N"] = "32"
    cp = q.OUT / "truth_32.json"; cache = json.loads(cp.read_text())
    jobs = []
    for i, k in enumerate(K):
        ch = C["nominal"][i]; h = q._hash(ch)
        if h not in cache:
            jobs.append((h, ((i, 7, 0), C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i], ch))); cache[h] = None
    print(len(jobs), "jobs", flush=True)
    with mp.get_context("spawn").Pool(24) as pool:
        for h, flags in pool.imap_unordered(q._truth_one, jobs):
            cache[h] = sum(flags) / len(flags)
    cp.write_text(json.dumps(cache)); print("done", flush=True)
