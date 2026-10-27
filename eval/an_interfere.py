"""Exploratory (user 2026-10-27: analyse the interference): at every 0.2 s check of 1x episodes, re-run the always-on
edit's own test for the POLICY's action (cbf_filter.lsf_edit: 8 noisy one-step imaginations, matched 1x noise, value
value_lf_alea.pt, pass iff min_j V(z'_j) > V(z, g) / 2) and record why it would trigger, the state's fragility margin
l_hat(z) (margin_lf.pt) and the phase.
  runs: 'filter' = the N1M always-on edit (paired_lfc000_n1mdev; states it visited, with its real edits)
        'policy' = the 1x policy alone (paired_record_n1dev; the states the policy visits by itself)
usage: python3 eval/an_interfere.py <n_episodes>  -> results/jenga/deep_dive/interfere_rows.npz"""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "1"
os.environ["LSF_ALEA"] = "8"; os.environ["LSF_ALEA_SCALE"] = "1"; os.environ["LSF_CLF"] = "margin_lf.pt"
ROOT = "/home/corey/wksp/stable_lyap_filters"; sys.path[:0] = [ROOT + "/eval", ROOT + "/src"]
os.environ["CBF_WM"] = ROOT + "/results/program_b/oil_v1/model/model_OIL.pt"
import cem_filter as cf, inspector_data as insp, lsf as L, uncertainty_monitor_dev as um

mon = cf.Monitor(); torch = mon.torch; mdl = mon.model
vf, noise, clf = L.ValueFn("value_lf_alea.pt"), L.Noise(), L.load_clf("cuda"); n = 8
snip = np.load(um.BENCH)["snippets"]
NEP = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def check(sim, cmds):
    with torch.no_grad():
        z = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()[None]
        g = torch.as_tensor(np.r_[sim.target, cmds[0][3]][None], dtype=torch.float32, device="cuda")
        a = torch.as_tensor(np.asarray(cmds, np.float32).reshape(1, 8), device="cuda")
        zs = noise.starts(z.expand(n, *z.shape[1:]).contiguous()); off = noise.actions(a, n)[:, 0] - a
        act = mdl._act((a[:, None] + off[None]).reshape(n, 8))
        r = mdl.net(zs, torch.zeros_like(zs), act, torch.zeros(n, device="cuda"))
        G2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1).repeat(n, 1)
        Vn = vf.V((zs + r * mdl.stats.res_std).float(), G2).cpu().numpy()
        return float(vf.V(z, g)), float(clf(z, g)), Vn


rows = []
for tag, run in (("filter", "paired_lfc000_n1mdev"), ("policy", "paired_record_n1dev")):
    for ep in range(15000, 15000 + NEP):
        S = dict(np.load(f"{ROOT}/results/program_b/policy_rollouts/{run}/ep{ep}.npz"))
        nrng = np.random.default_rng([20261502, ep]); noises = [snip[nrng.integers(len(snip))] * 1.0 for _ in range(int(S["chunk_index"][-1]) + 1)]
        noise.rng = np.random.default_rng([20261901, ep])
        sim = insp._sim(); closed_once = False
        for d in range(len(S["chunk_index"])):
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))); k = int(S["chunk_index"][d])
            nom, exe = S["nominal"][d], S["executed"][d]
            for j in range(8):
                if j % 2 == 0:
                    vz, lz, Vn = check(sim, nom[j:j + 2])
                    closed = bool(nom[j, 3] > 0); closed_once = closed_once or closed
                    phase = 1 if closed else (2 if closed_once else 0)          # 0 approach, 1 holding, 2 after a release
                    edited = float(np.abs(exe[j:j + 2, :3] - nom[j:j + 2, :3]).max() * 1000)
                    tilt = sim.block_diagnostics()["tilt"]
                    rows.append((0 if tag == "filter" else 1, ep, k, j, phase, vz, lz, Vn.min(), Vn.mean(), np.median(Vn), edited, float(max(tilt[1:]))))
                a = np.array(exe[j], np.float32); a[:3] -= noises[k][j]; sim.execute(a)
        sim.close(); print(tag, ep, len(rows), flush=True)
np.savez(ROOT + "/results/jenga/deep_dive/interfere_rows.npz", rows=np.array(rows, float),
         cols=np.array(["run", "ep", "k", "j", "phase", "Vz", "l_frag", "Vnext_min", "Vnext_mean", "Vnext_med", "edit_mm", "neigh_tilt"]))
print("saved", flush=True)
