"""Calibration / recall of the closed-loop hybrid (HYB_CF: model_CF + residual flow head) on the same 1,575 plans as
eval/wm_reliability.py (75 risky dev decisions x 21 plans; real risk cached). 64 hybrid futures per plan from the TRUE
start latent (the head samples the residual), actions = H15 horizon with the 64 execution-noise snippets (x1, as the
deterministic monitor at s = 1). Imagined topple fraction q = share of futures in the fork test's group with larger
change away from the hand (0 if no fork)."""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q
from scipy.stats import spearmanr
import torch, cem_filter as cf, cbf_barrier as cb, cbf_filter as cbff, phase2_hybrid as hy, step3_world_model as wm
import lookahead_check as la
C, K = q.load_cache()
T = json.loads((q.OUT / "truth_32.json").read_text())
rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
MD = q.ROOT / "results/program_b/cf_v1/model"
s = torch.load(MD / "model_HYB_CF.pt", map_location="cuda")
head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
det = wm.WorldModel.load(MD / "model_CF.pt"); det.net.eval()
snip = np.load(q.ROOT / "results/jenga/bench/jenga_bench.npz")["snippets"]
Q, R, NF = [], [], []
for n_, r in enumerate(rows):
    i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
    P = np.asarray(r["plans"], np.float32)
    for c in range(len(P)):
        acts = np.stack([la.horizon_actions(P[c], "H15", la.noise_for(snip, j)) for j in range(64)])
        a = det._act(torch.as_tensor(acts, device="cuda").reshape(64, wm.STEPS, -1))
        g = torch.Generator(device="cuda").manual_seed(5000 + 100 * i + c)
        e = hy.sample(head, det, z0[None].expand(64, -1, -1, -1).contiguous(), a, s["residual_scale"], g)   # (64, 2, 28, 28, 8)
        e10, e30 = e[:, 0].reshape(1, 64, -1), e[:, 1].reshape(1, 64, -1)
        ok10, _, _ = cf.split_test_gpu(e10); ok30, lab, _ = cf.split_test_gpu(e30)
        fork = bool((ok10 & ok30).item())
        pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        o_j = (((e[:, 1] - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum())
        L = lab[0].bool()
        if fork and not (L.all() or (~L).all()):
            bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
            Q.append(float(bad.float().mean()))
        else:
            Q.append(0.0)
        NF.append(not fork); R.append(T[q._hash(P[c][1:9])])
    if n_ % 15 == 0:
        print("decision", n_ + 1, flush=True)
Q, R, NF = np.array(Q), np.array(R), np.array(NF)
edges = [0, 1e-9, 0.1, 0.2, 0.35, 0.5, 1.01]; bins = []
for a_, b_ in zip(edges[:-1], edges[1:]):
    mk = (Q >= a_) & (Q < b_)
    if mk.any(): bins.append({"imagined": f"[{a_:.2f},{b_:.2f})", "n": int(mk.sum()), "mean_imagined": round(float(Q[mk].mean()), 3), "real_rate": round(float(R[mk].mean()), 3)})
res = {"model": "HYB_CF", "plans": len(R), "ECE": round(sum(b["n"] * abs(b["mean_imagined"] - b["real_rate"]) for b in bins) / len(R), 3),
       "spearman_all": round(float(spearmanr(Q, R).correlation), 3), "no_fork_share": round(float(NF.mean()), 3),
       "real_rate_when_no_fork": round(float(R[NF].mean()), 3), "real_rate_when_fork": round(float(R[~NF].mean()), 3), "bins": bins}
json.dump(res, open(q.OUT / "reliability_hyb.json", "w"), indent=1); print(json.dumps(res, indent=1))
