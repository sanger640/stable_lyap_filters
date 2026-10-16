"""Ensemble check (4 deterministic members, results/program_b/ens_v1): on the 75 risky dev decisions x 13 plans (policy
plan + 12 policy samples; real risk cached), each member imagines 64 futures per plan (perception-error starts via G +
execution noise, scale 1) -> imagined topple fraction q_k. Ensemble mean q-bar, disagreement sd_k(q).
Questions: (1) does q-bar rank a decision's plans by real risk better than one model (within-decision Spearman, real
risk of the min-q-bar pick)? (2) is disagreement larger where a single model is wrong (|q_k - real| vs sd)?"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb, step3_world_model as wm
ENS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/program_b/ens_v1")
C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
mon = cf.Monitor()
members = sorted(ENS.glob("model_ENS*.pt"))
Qall = np.zeros((len(members), len(rows), 13)); R = np.zeros((len(rows), 13))
for m_, mp in enumerate(members):
    mon.model = wm.WorldModel.load(mp); mon.model.net.eval()
    for n_, r in enumerate(rows):
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        P = np.asarray(r["plans"], np.float32)[:13]
        R[n_] = [T[q._hash(p[1:9])] for p in P]
        bar = q.Barrier(mon, z0, 1000 + i, 64, 1.0)
        with torch.no_grad():
            e10, e30 = bar.endpoints(torch.as_tensor(P, device="cuda"))
        ok10, _, _ = cf.split_test_gpu(e10); ok30, lab, _ = cf.split_test_gpu(e30)
        for c in range(len(P)):
            pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
            msk = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            o_j = (((e30[c].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1) * msk).sum((1, 2)) / msk.sum())
            L = lab[c].bool()
            if bool(ok10[c] & ok30[c]) and not (L.all() or (~L).all()):
                bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
                Qall[m_, n_, c] = float(bad.float().mean())
        torch.cuda.empty_cache()
    print("member", m_, "done", flush=True)
qbar, sd = Qall.mean(0), Qall.std(0)
def within(S):
    sp = [spearmanr(S[d], R[d]).correlation for d in range(len(rows)) if np.ptp(S[d]) > 0 and np.ptp(R[d]) > 0]
    pick = R[np.arange(len(rows)), np.argmin(S + 1e-6 * np.arange(13), 1)]
    return {"spearman": round(float(np.mean(sp)), 3), "pick_real_risk": round(float(pick.mean()), 4)}
res = {"members": len(members), "random_sample": round(float(R[:, 1:].mean()), 4), "oracle": round(float(R.min(1).mean()), 4),
       "policy_plan": round(float(R[:, 0].mean()), 4),
       "single_members": [within(Qall[k]) for k in range(len(members))], "ensemble_mean": within(qbar),
       "ensemble_mean_plus_sd": within(qbar + sd)}
err = np.abs(Qall[0] - R).ravel(); s_ = sd.ravel()
res["disagreement_vs_error"] = {"spearman_sd_vs_abs_error_member0": round(float(spearmanr(s_, err).correlation), 3),
                                "mean_abs_error_low_sd": round(float(err[s_ <= np.median(s_)].mean()), 3),
                                "mean_abs_error_high_sd": round(float(err[s_ > np.median(s_)].mean()), 3)}
np.savez(q.OUT / f"ens_scores_{ENS.name}.npz", Q=Qall, R=R)
json.dump(res, open(q.OUT / f"ens_check_{ENS.name}.json", "w"), indent=1); print(json.dumps(res, indent=1))
