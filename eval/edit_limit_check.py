"""Edit-limit check (exploratory, offline; user 2026-10-26: why a 1 cm edit limit?). For the selected pure edit (lfck000,
GATE3 dev episodes 15000-15099), replay every gate-open chunk to the state of each 0.2 s check (saved decision state + the
run's executed commands and execution noise, as eval/filter_videos.py) and re-run the filter's own barrier test
(cbf_filter.lsf_edit: 8 noisy one-step imaginations in model_OIL, value_lf_alea.pt, pass iff min_j V(z'_j) >= V(z, g) / 2)
on a 5 mm grid of targets: both commands of the model step shifted by the same (dx, dy, dz), |d|_inf <= 2 cm (9^3 = 729).
Fresh noise draws, common to all candidates. For the checks where the policy's own action fails (the edit is triggered):
  feasible_1cm  some target with |d|_inf <= 1 cm passes (inside the current limit)
  only_2cm      none within 1 cm, some within 2 cm (the limit binds)
  none          no grid target passes
Grid resolution 5 mm, so 'none within 1 cm' can miss narrow passing regions the gradient edit would find (upper bound on binding).
Output: results/jenga/edit_limit_check.json (+ _rows.npz)
usage: python3 eval/edit_limit_check.py [max_episodes]
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "2"
os.environ["CBF_WM"] = str(Path(__file__).resolve().parent.parent / "results/program_b/oil_v1/model/model_OIL.pt")
os.environ["LSF_ALEA"] = "8"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUN = ROOT / "results/program_b/policy_rollouts/paired_lfck000_gate3dev"
G = np.round(np.arange(-0.02, 0.0201, 0.005), 4)


def main():
    import cem_filter as cf
    import inspector_data as insp
    import lsf as L
    import step3_world_model as wm
    import uncertainty_monitor_dev as um
    mon = cf.Monitor(); torch = mon.torch; mdl = mon.model
    vf, noise = L.ValueFn("value_lf_alea.pt"), L.Noise(); n = 8
    D = np.array([[x, y, z] for x in G for y in G for z in G], np.float32)
    box1 = np.abs(D).max(1) <= 0.01 + 1e-6
    Dt = torch.as_tensor(np.concatenate([D, np.zeros((len(D), 1)), D, np.zeros((len(D), 1))], 1).astype(np.float32), device="cuda")
    snippets = np.load(um.BENCH)["snippets"]
    sim = insp._sim()

    def margins(cmds):
        with torch.no_grad():
            z = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()[None]
            g = torch.as_tensor(np.r_[sim.target, cmds[0][3]][None], dtype=torch.float32, device="cuda")
            a_pol = torch.as_tensor(np.asarray(cmds, np.float32).reshape(1, 8), device="cuda")
            zs = noise.starts(z.expand(n, *z.shape[1:]).contiguous())
            off = noise.actions(a_pol, n)[:, 0] - a_pol
            thr = 0.5 * float(vf.V(z, g)); out = []
            for A in torch.split(a_pol + Dt, 243):
                c = len(A); Z = zs.repeat(c, *[1] * (zs.dim() - 1))
                act = mdl._act((A[:, None] + off[None]).reshape(c * n, 8))
                r = mdl.net(Z, torch.zeros_like(Z), act, torch.zeros(len(Z), device="cuda"))
                G2 = torch.cat([A[:, 4:7], A[:, 7:8]], 1).repeat_interleave(n, 0)
                out.append(vf.V((Z + r * mdl.stats.res_std).float(), G2).view(c, n).min(1).values)
        return torch.cat(out).cpu().numpy() - thr
    assert mdl.kind == "deterministic"
    i0 = int(np.flatnonzero((np.abs(D) < 1e-6).all(1))[0])          # the policy's own action (d = 0)
    eps = sorted(int(p.stem[2:]) for p in RUN.glob("ep*.npz"))[:int(sys.argv[1]) if len(sys.argv) > 1 else None]
    rows = []
    for ep in eps:
        with np.load(RUN / f"ep{ep}.npz") as f:
            S = {k: f[k] for k in f.files}
        log = {c["chunk"]: c for c in json.loads((RUN / f"ep{ep}.json").read_text())["log"]}
        nrng = np.random.default_rng([20261502, ep])
        noises = [snippets[nrng.integers(len(snippets))] * 2.0 for _ in range(int(S["chunk_index"][-1]) + 1)]
        noise.rng = np.random.default_rng([20261888, ep]); mon.rng = np.random.default_rng([20261889, ep])
        for d in range(len(S["chunk_index"])):
            k = int(S["chunk_index"][d])
            if not log.get(k, {}).get("irr_gate"):
                continue
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
            nom, exe = np.asarray(S["nominal"][d], np.float32), np.asarray(S["executed"][d], np.float32)
            for j in range(8):
                if j % 2 == 0:
                    m = margins(nom[j:j + 2])
                    ok = m >= 0
                    small = float(np.abs(D[ok]).max(1).min() * 1000) if ok.any() else np.nan     # smallest passing shift (mm, |d|_inf)
                    rows.append((ep, k, j, m[i0], m[box1].max(), m.max(), float(np.abs(exe[j:j + 2, :3] - nom[j:j + 2, :3]).max() * 1000), small))
                a = exe[j].copy(); a[:3] -= noises[k][j]; sim.execute(a)
        print("ep", ep, len(rows), flush=True)
    R = np.array(rows, float)
    trig = R[:, 3] < 0
    f1, f2 = R[:, 4] >= 0, R[:, 5] >= 0
    out = {"episodes": len(eps), "checks": int(len(R)), "triggered": int(trig.sum()),
           "feasible_1cm": int((trig & f1).sum()), "only_2cm": int((trig & ~f1 & f2).sum()), "none": int((trig & ~f2).sum()),
           "share_of_triggered": {"feasible_1cm": round(float((trig & f1).mean() / max(trig.mean(), 1e-9)), 3),
                                  "only_2cm": round(float((trig & ~f1 & f2).mean() / max(trig.mean(), 1e-9)), 3),
                                  "none": round(float((trig & ~f2).mean() / max(trig.mean(), 1e-9)), 3)},
           "run_edited_steps_among_triggered": int((trig & (R[:, 6] > 0.5)).sum()),
           "smallest_passing_shift_mm_triggered": {q: float(np.nanpercentile(R[trig, 7], v)) for q, v in (("median", 50), ("p90", 90))} if trig.any() else None,
           "smallest_passing_shift_mm_hist_triggered": {str(int(v)): int((R[trig, 7] == v).sum()) for v in (5, 10, 15, 20)},
           "grid": "5 mm, |d|_inf <= 2 cm, both commands shifted together, gripper the policy's", "noise": "fresh draws, 8, common to candidates",
           "status": "exploratory (not pre-registered)"}
    (ROOT / "results/jenga/edit_limit_check.json").write_text(json.dumps(out, indent=1))
    np.savez(ROOT / "results/jenga/edit_limit_check_rows.npz", rows=R)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
