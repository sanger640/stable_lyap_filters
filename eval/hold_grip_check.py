"""Does the filter's hold reference (gripper command 0.0, which the WM base data reads as 'open') differ from a hold that
keeps the gripper command (= policy's step-0 command)? Closed-gripper bench2 dev + holdout decisions. Exploratory."""
import json, os, sys, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = "/home/corey/wksp/stable_lyap_filters"
os.environ.setdefault("CBF_WM", ROOT + "/results/program_b/oil_v1/model/model_OIL.pt")
sys.path[:0] = [ROOT + "/eval", ROOT + "/src", ROOT + "/libero"]; os.chdir(ROOT)
import torch, bench2 as b2, cbf_barrier, cbf_filter as cbff, cem_filter as cf, inspector_data as insp
cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); rows = []
for s in ("dev", "holdout"):
    C, _ = b2.load(s); cache = json.loads((b2.OUT / f"truth_{s}.json").read_text())
    for i in range(len(C["gripper"])):
        if not C["gripper"][i] < 50: continue
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        g = float(C["nominal"][i][0, 3])
        plans = [C["plan16"][i]] + [np.repeat(np.r_[sim.target, v][None].astype(np.float32), 16, 0) for v in (0.0, g, 1.0)]
        mon.rng = np.random.default_rng([20261127, i])
        h, o = cbf.score(sim, z0, np.stack(plans), 64)
        tr_pol = cache.get(b2._hash(i, np.asarray(C["nominal"][i], np.float32)))
        tr_hold = cache.get(b2._hash(i, b2.hold_chunk(C, i)))
        rows.append({"set": s, "i": i, "g": g, "h": [float(x) for x in h], "o": [float(x) for x in o], "truth_policy": tr_pol, "truth_hold": tr_hold})
        print(s, i, np.round(h, 2), flush=True)
json.dump(rows, open(ROOT + "/results/jenga/deep_dive/hold_grip_rows.json", "w"))
R = rows; H = np.array([r["h"] for r in R]); print("n", len(R), "g>0.9 share", np.mean([r["g"] > 0.9 for r in R]))
for j, n in enumerate(("plan", "hold0", "hold_keep", "hold+1")):
    print(n, "mean h %.2f  h<2 %.3f" % (H[:, j].mean(), (H[:, j] < 2).mean()))
t0 = H[:, 0] < np.minimum(H[:, 1], 2) - 1e-9; tk = H[:, 0] < np.minimum(H[:, 2], 2) - 1e-9
red = np.array([r["truth_policy"][1] if r["truth_policy"] else np.nan for r in R])
print("trigger g0 %.3f  keep %.3f  only_keep %d only_g0 %d" % (t0.mean(), tk.mean(), (tk & ~t0).sum(), (t0 & ~tk).sum()))
for nm, m in (("only_keep", tk & ~t0), ("both", tk & t0), ("neither", ~tk & ~t0)):
    print(nm, int(m.sum()), "policy red risk %.3f" % np.nanmean(red[m]) if m.any() else "")
