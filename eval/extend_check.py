"""Would more time rescue the optimiser's 'never picked up' episodes (dev10)? Continue each from its last saved decision
(executing that decision's chunk with the run's noise) with the POLICY ALONE for EXTRA more chunks (fresh noise draws),
tracking the red block's lift. -> results/jenga/extend_check_dev10.json
usage: DIFFUSION_POLICY_DIR=... dp_venv/bin/python eval/extend_check.py [extra_chunks=40]"""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main(extra=40):
    import torch
    import inspector_data as insp, plausibility as pl, policy_rollout as pr
    import uncertainty_monitor_dev as um
    extra = int(extra); snippets = np.load(um.BENCH)["snippets"]
    lab = json.loads((ROOT / "results/program_b/dev10_videos/labels.json").read_text())["cbfmoil"]
    eps = sorted(int(e) for e, l in lab.items() if l == "M")
    pol = pr.load_policy(); sim = insp._sim(); cams = pr.Cameras(sim); out = []
    for ep in eps:
        with np.load(ROOT / f"results/program_b/policy_rollouts/paired_cbfmoil_dev10/ep{ep:03d}.npz") as f:
            S = {k: f[k] for k in f.files}
        noises = pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1)
        sim.restore((S["state"][0], S["ctrl"][0], S["target"][0], float(S["gripper"][0]))); z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
        d = len(S["chunk_index"]) - 1
        hist = pl.obs_hist(sim, cams, S, d, noises)               # ends restored at decision d
        k = int(S["chunk_index"][d])
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a); hist.append(cams.obs())
        torch.manual_seed(ep); rng = np.random.default_rng([7, ep]); lift = []; tilt = []
        for c in range(extra):
            chunk = pr.policy_chunk(pol, hist)
            nz = snippets[rng.integers(len(snippets))] * 2.0
            for j, a in enumerate(chunk):
                a = np.asarray(a, np.float32).copy(); a[:3] -= nz[j]; sim.execute(a); hist.append(cams.obs())
                t = sim.block_diagnostics()["tilt"]
                lift.append(float(sim.data.xpos[sim.tracked_block_ids[0]][2] - z0)); tilt.append((float(t[0]), float(np.max(t[1:]))))
        L = np.array(lift); T = np.array(tilt)
        i2 = int(np.argmax(L >= 0.02)) if (L >= 0.02).any() else None
        out.append({"ep": ep, "picked_up_in_extra_time": i2 is not None, "seconds_needed": round(i2 * 0.1, 1) if i2 is not None else None,
                    "max_lift_mm": round(float(L.max() * 1000), 1), "neighbour_toppled": bool(T[:, 1].max() >= 45), "red_toppled": bool(T[:, 0].max() >= 45)})
        print(out[-1], flush=True)
    summ = {"episodes": len(out), "extra_seconds": extra * 0.8, "picked_up": sum(r["picked_up_in_extra_time"] for r in out),
            "picked_up_without_any_topple": sum(r["picked_up_in_extra_time"] and not r["neighbour_toppled"] and not r["red_toppled"] for r in out)}
    (ROOT / "results/jenga/extend_check_dev10.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1)); print(json.dumps(summ))


if __name__ == "__main__":
    main(*sys.argv[1:])
