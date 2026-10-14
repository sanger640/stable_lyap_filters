"""T2 corrected data (results/program_b/t2_v2): robot-still unit test on H_T2 and a verification video
(as eval/verify_perturbation_video.py)."""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

T2V = ROOT / "results/program_b/t2_v2"


def unit_job(name):
    fam, seed, s = name.split("_"); seed, index = int(seed), int(s[1:])
    scene, sim, acts, starts = pdata.prepare(fam, seed)
    for i, a in enumerate(acts):
        if i == index:
            break
        sim.execute(a)
    pc, rc = ps.pooled_error_model(); snap = sim.snapshot(); start = sim.object_state(); n = len(scene["objects"])
    ev = 0
    for j in range(16):
        sim.restore(snap); sim.perturb(*ps.draw_error(n, pc, rc, pm.SEEDS["h2_futures"], pdata.FAMILY_CODE[fam], seed, index, j))
        hold = np.r_[sim.target, 0.0]
        for _ in range(38):
            sim.execute(hold)
        tr = sim.object_state()
        ev += bool(((np.linalg.norm(tr[:, :3] - start[:, :3], axis=1) >= 0.03) |
                    (np.abs(ps.tilt_deg(tr[:, 3:]) - ps.tilt_deg(start[:, 3:])) >= 45)).any())
    return name, ev


def main():
    import multiprocessing as mp
    names = sorted(p.stem for p in (T2V / "h2").glob("T2_*.npz"))
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.map(unit_job, names))
    out = {"states": len(res), "still_futures": 16 * len(res), "still_with_event": sum(res.values()),
           "states_with_still_event": sum(v > 0 for v in res.values()), "per_state": res}
    (ROOT / "results/jenga/t2_v2_unit_test.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_state"}), flush=True)
    # video: 4 forks + 2 safe from H_T2 (corrected)
    import verify_perturbation_video as vv
    forks, safe = [], []
    for nm in names:
        with np.load(T2V / "h2" / f"{nm}.npz") as f:
            k = int(f["harm"].sum())
        (forks if 2 <= k <= 62 else safe if k == 0 else []).append(nm)
    rng = np.random.default_rng(5)
    vv.SCENES = [forks[i] for i in rng.permutation(len(forks))[:4]] + [safe[i] for i in rng.permutation(len(safe))[:2]]
    vv.OUTV = ROOT / "results/program_b/t2_fix_verification.mp4"
    vv.SUMMARY = ROOT / "results/jenga/t2_fix_video_summary.json"
    vv.main()


if __name__ == "__main__":
    main()
