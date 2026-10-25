"""Equivalence + timing check for CERT_PRUNE (eval/cbf_filter.py score, eval/opt2.py certify): the pruned certificate skips
roll-outs that cannot change the decision (scales above the first split; certifier models after a plan already failed).
Exact in logic; bf16 batch numerics may differ. On the first N cert_recheck cases (exact saved states): the psel decision
with CERT_PRUNE off and on, same policy samples and certificate seed; compares ok vectors, model-0 h / o, kind / choice,
the rng state after the call, and seconds.
usage: N=24 python3 eval/prune_check.py -> results/jenga/deep_dive/prune_check.json
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cert_recheck as cr  # noqa: E402
import grip_bench as gb  # noqa: E402


def main():
    import torch
    import plausibility as pl
    import fix_counterfactual as fc
    import opt3
    import policy_rollout as pr
    import cbf_filter as cbff
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    mon, cbf, certs = gb.setup(); pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="pc_")))); cams = pr.Cameras(sim)
    rows = []
    for stratum, e, d in cr.cases()[::max(1, len(cr.cases()) // int(os.environ.get("N", 24)))][:int(os.environ.get("N", 24))]:
        with np.load(gb.D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        noises = pl.episode_noises(e, gb.LAST + 2, scale=fc.NOISE)
        sim.restore((S["state"][d - 1], S["ctrl"][d - 1], S["target"][d - 1], float(S["gripper"][d - 1])))
        hist = []
        for j, a in enumerate(S["executed"][d - 1]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[int(S["chunk_index"][d - 1])][j]; sim.execute(a)
            if j >= 6:
                hist.append(cams.obs())
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        mon.rng = np.random.default_rng([cr.SEED, e, d])
        z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
        p16 = np.asarray(S["plan16"][d], np.float32).copy(); p16[0, :3] = sim.target
        hold16 = np.repeat(np.r_[sim.target, p16[0, 3]][None], 16, 0).astype(np.float32)
        h, o = cbf.score(sim, z0, np.stack([p16, hold16]), 64); h_now = float(h[1])
        P = opt3.Problem(mon, cbf, sim, z0, p16, h_now, certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
        torch.manual_seed(int(cr.SEED + 7 + 1000 * e + d))
        samples = np.asarray(pr.policy_chunk(pol, hist, n=16), np.float32)
        res = {}
        for name, flag in (("full", False), ("pruned", True)):
            cbff.CFG["PRUNE"] = flag; mon.rng = np.random.default_rng([cr.SEED, e, d, 1])
            torch.cuda.synchronize(); t0 = time.time()
            ok, H, O = P.certify(list(samples) + [hold16])
            torch.cuda.synchronize()
            res[name] = {"ok": ok.tolist(), "h": H.tolist(), "o": O.tolist(), "sec": round(time.time() - t0, 2),
                         "rng_next": int(mon.rng.integers(2 ** 31))}
        cbff.CFG["PRUNE"] = False
        F, Q = res["full"], res["pruned"]
        row = {"stratum": stratum, "ep": e, "d": d, "ok_equal": F["ok"] == Q["ok"], "h_equal": F["h"] == Q["h"],
               "o_maxabs": float(np.max(np.abs(np.array(F["o"]) - np.array(Q["o"])))), "rng_equal": F["rng_next"] == Q["rng_next"],
               "n_ok": int(np.sum(F["ok"][:16])), "sec_full": F["sec"], "sec_pruned": Q["sec"]}
        rows.append(row); print(row, flush=True)
    out = {"cases": len(rows), "ok_equal": sum(r["ok_equal"] for r in rows), "h_equal": sum(r["h_equal"] for r in rows),
           "rng_equal": sum(r["rng_equal"] for r in rows), "o_maxabs": max(r["o_maxabs"] for r in rows),
           "sec_full_median": float(np.median([r["sec_full"] for r in rows])),
           "sec_pruned_median": float(np.median([r["sec_pruned"] for r in rows])), "rows": rows}
    (ROOT / "results/jenga/deep_dive/prune_check.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})
    cams.close(); sim.close(); os._exit(0)


if __name__ == "__main__":
    main()
