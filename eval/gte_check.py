"""Object-free start sampler for the filter: Gauss-TE (stationary Gaussian field added to the latent; Phase 2 P2-0b
estimator, eval/phase2_p0.fit_te) refit on the closed-loop training states (z0k - z0), vs the object-centric G.

fit    -> results/program_b/closedloop_v1/model/Gauss_TE_cl.pt
check  -> on the 361 cached dev3A decisions (real risk of the policy chunk cached, 32 futures): barrier h of the policy
          plan and of holding still with G vs Gauss-TE starts (same 4 scales, same execution noise, model_US);
          AUC for real risk >= 2/32, and caught / false alarms of the final rule h < min(h_now, log2 3).
usage: python3 eval/gte_check.py fit | check
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CBF_WM", os.path.abspath("results/program_b/closedloop_v1/model/model_US.pt"))
ROOT = Path(__file__).resolve().parent.parent
GTE = ROOT / "results/program_b/closedloop_v1/model/Gauss_TE_cl.pt"


def fit():
    import closedloop_train as ct
    import phase1d_models as p1d
    import phase2_p0 as p0
    import step3b_models as sm
    sm.DATA = ROOT / "results/program_b/threeblock_v3"
    ct._patch()
    states = p1d.load([ROOT / "results/program_b/closedloop_v1/train_selected"], proprio=False)
    p0.OUT = GTE.parent
    te = p0.fit_te(states)
    (GTE.parent / "Gauss_TE.pt").rename(GTE)
    print("fitted on", len(states), "states; lag-0 cov trace", float(te["lag0_cov"].diagonal().sum()), flush=True)


class GaussStarts:
    def __init__(self):
        import torch
        self.torch = torch; self.root = torch.load(GTE)["root"].to("cuda")

    def __call__(self, z0, n, scale, seed):
        torch = self.torch
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        w = torch.randn(n, 28, 28, 8, device="cuda", generator=g).double()
        f = torch.fft.fft2(w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.einsum("abij,nabj->nabi", self.root, f)
        field = torch.fft.ifft2(x.permute(0, 3, 1, 2)).real.permute(0, 2, 3, 1).float()
        return z0[None].float() + scale * field


def barrier(mon, z0, plans, starts_fn, seed, n=64):
    """h per plan (C,) with starts_fn(scale) -> (n,28,28,8) starts; mirrors cbf_filter.CBF.score's h."""
    import torch
    import cbf_barrier as cb
    import cem_filter as cf
    import cbf_filter as cbff
    import step3_world_model as wm
    C = len(plans); split = np.zeros((C, len(cb.SCALES)), bool)
    for si, s in enumerate(cb.SCALES):
        st = starts_fn(s)
        acts = []
        for p in plans:
            base = cbff.horizon(p, cbff.CFG["N_LOOK"])
            for j in range(n):
                a = base.copy(); noise = np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
                a[:cbff.CFG["N_LOOK"], :3] -= noise; acts.append(a)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n, wm.STEPS, -1)
        z = st.float().repeat(C, 1, 1, 1); ends = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = mon.model.step(z, pairs[:, t]).float()
                if t in (wm.HOLD10, wm.HOLD30):
                    ends[t] = z.reshape(C, n, -1)
        ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
        split[:, si] = (ok10 & ok30).cpu().numpy()
    h = np.full(C, np.log2(cb.SCALES[-1]))
    for c in range(C):
        hit = np.flatnonzero(split[c])
        if len(hit):
            h[c] = np.log2(cb.SCALES[hit[0]])
    return h


def check():
    import torch
    import cbf_qp as q
    import cem_filter as cf
    from sklearn.metrics import roc_auc_score
    C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
    mon = cf.Monitor(seed=777); gs = GaussStarts()
    rows = []
    for i, k in enumerate(K):
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        hold = np.repeat(np.r_[C["target"][i], 0.0][None], 16, 0).astype(np.float32)
        plans = np.stack([C["plan16"][i].astype(np.float32), hold])
        seed = 5000 + i

        def g_starts(s):
            d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                           for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(64))])
            with torch.no_grad():
                return mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
        hG = barrier(mon, z0, plans, g_starts, seed)
        hT = barrier(mon, z0, plans, lambda s: gs(z0, 64, s, seed), seed)
        rows.append({"risk": T[q._hash(C["nominal"][i])], "hG": hG.tolist(), "hT": hT.tolist()})
        if i % 60 == 0:
            print("decision", i, flush=True)
    risk = np.array([r["risk"] for r in rows]); y = risk >= 2 / 32; hs = np.log2(3.0)
    def rate(flag):
        flag = np.asarray(flag, bool)
        return {"flagged": int(flag.sum()), "caught_risky": round(float(flag[y].mean()), 3), "false_on_safe": round(float(flag[~y].mean()), 3)}
    res = {"decisions": len(rows), "risky": int(y.sum())}
    for name, key in (("G_object_centric", "hG"), ("GaussTE_object_free", "hT")):
        hp = np.array([r[key][0] for r in rows]); hn = np.array([r[key][1] for r in rows])
        res[name] = {"auc_minus_h": round(float(roc_auc_score(y, -hp)), 3),
                     "final_rule_h_lt_min(hnow,log2_3)": rate(hp < np.minimum(hn, hs)), "first_rule_h_lt_min(hnow,1)": rate(hp < np.minimum(hn, 1.0))}
    (q.OUT / "gte_check.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"fit": fit, "check": check}[sys.argv[1]]()
