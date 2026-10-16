"""Learned barrier B(z0, plan): one smooth, calibrated risk function used both to optimise and to accept a plan.

Inputs (latent only, differentiable through the world model): statistics of the imagined futures of the plan
(16 futures, execution noise, start sampler G): per-future latent change away from the hand at hold 10 and hold 30
(mean, CVaR_0.25, std, max) and the smooth fork margin at hold 10 / 30 (cbf_qp.Barrier.margin). 10 features.
Output: predicted real topple fraction (logistic regression, L2, weighted by 32 futures). Labels: real simulator,
32 futures (training only; the deployed barrier sees latents only).
Data: 75 risky dev decisions x 21 plans (policy plan, 12 samples, 8 nudges) + the policy plan at the other cached
decisions. Cross-validated by decision (5 folds over the 75 risky ones; calm decisions always in training), so every
test decision is scored by a barrier fitted without it.

usage: python3 eval/learned_barrier.py extract | fit
"""
import json, os, sys
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl")
import cbf_qp as q

OUT = q.OUT / "learned_barrier"
N_FUT = 16


def features(bar, plans, z0, site):
    """plans (C,16,4) torch -> (C, 10) torch, differentiable."""
    import torch
    import cbf_barrier as cb
    e10, e30 = bar.endpoints(plans)
    feats = []
    for c in range(len(plans)):
        pts = np.vstack([site[None], plans[c][1:16, :3].detach().cpu().numpy()])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        row = []
        for e in (e10, e30):
            o = ((e[c].reshape(-1, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()
            k = max(1, int(np.ceil(0.25 * len(o))))
            row += [o.mean(), o.topk(k).values.mean(), o.std(), o.max()]
        feats.append(torch.stack(row))
    F = torch.stack(feats)
    mg = torch.stack([bar.margin(e10), bar.margin(e30)], 1)
    return torch.cat([F, mg], 1)


def extract():
    import torch
    import cem_filter as cf
    C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
    rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
    risky = {r["i"]: np.asarray(r["plans"], np.float32) for r in rows}
    mon = cf.Monitor()
    X, Y, D = [], [], []
    for i in range(len(C["z0"])):
        if i in risky:
            P = risky[i]
        else:
            P = C["plan16"][i][None].astype(np.float32)
        lab = [T.get(q._hash(p[1:9])) for p in P]
        keep = [j for j, v in enumerate(lab) if v is not None]
        if not keep:
            continue
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        bar = q.Barrier(mon, z0, 1000 + i, N_FUT, 1.0)
        with torch.no_grad():
            f = features(bar, torch.as_tensor(P[keep], device="cuda"), z0, C["site"][i]).cpu().numpy()
        X.append(f); Y += [lab[j] for j in keep]; D += [i] * len(keep)
        if i % 50 == 0:
            print("decision", i, flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "data.npz", X=np.concatenate(X), y=np.array(Y), d=np.array(D), risky=np.array(sorted(risky)))
    print("extracted", len(Y), "plans from", len(set(D)), "decisions")


def _fit(X, y, l2=1e-2):
    """Logistic regression on standardised features, fraction labels (32 trials each). Returns dict of numpy params."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float64); yt = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500)

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return {"mu": mu, "sd": sd, "w": w.detach().numpy(), "b": float(b.detach())}


def predict(p, X):
    return 1 / (1 + np.exp(-(((X - p["mu"]) / p["sd"]) @ p["w"] + p["b"])))


def fit():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    d = np.load(OUT / "data.npz"); X, y, D, risky = d["X"], d["y"], d["d"], d["risky"]
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(risky), 5)
    pred = np.full(len(y), np.nan); models = []
    for k, test in enumerate(folds):
        tr = ~np.isin(D, test)
        p = _fit(X[tr], y[tr]); models.append({"test": test.tolist(), **{a: (v.tolist() if hasattr(v, "tolist") else v) for a, v in p.items()}})
        te = np.isin(D, test); pred[te] = predict(p, X[te])
    m = np.isin(D, risky)
    # within-decision pick: plan with the lowest B among the 21 (vs the mean_o feature alone, vs random)
    pick_B, pick_o, rand = [], [], []
    for i in risky:
        s = D == i
        pick_B.append(y[s][np.argmin(pred[s])]); pick_o.append(y[s][np.argmin(X[s, 4])]); rand.append(y[s].mean())
    res = {"plans": int(len(y)), "risky_plans": int(m.sum()),
           "cv_spearman_B": round(float(spearmanr(pred[m], y[m]).correlation), 3),
           "spearman_mean_o30": round(float(spearmanr(X[m, 4], y[m]).correlation), 3),
           "cv_auc_B (risk>=2/32)": round(float(roc_auc_score(y[m] >= 2 / 32, pred[m])), 3),
           "cv_calibration": {"mean_pred": round(float(pred[m].mean()), 3), "mean_real": round(float(y[m].mean()), 3)},
           "pick_lowest_B": round(float(np.mean(pick_B)), 4), "pick_lowest_mean_o30": round(float(np.mean(pick_o)), 4),
           "random_plan": round(float(np.mean(rand)), 4)}
    full = _fit(X, y)
    json.dump({"cv": res, "folds": models, "full": {a: (v.tolist() if hasattr(v, "tolist") else v) for a, v in full.items()}},
              open(OUT / "barrier.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"extract": extract, "fit": fit}[sys.argv[1]]()


class Barrier:
    """Torch version of the fitted barrier: B(features) = sigmoid(w . (f - mu) / sd + b), differentiable."""

    def __init__(self, params):
        import torch
        self.mu = torch.as_tensor(params["mu"], device="cuda", dtype=torch.float32)
        self.sd = torch.as_tensor(params["sd"], device="cuda", dtype=torch.float32)
        self.w = torch.as_tensor(params["w"], device="cuda", dtype=torch.float32); self.b = float(params["b"])

    def __call__(self, F):
        import torch
        return torch.sigmoid(((F - self.mu) / self.sd) @ self.w + self.b)


def load(decision=None):
    """The barrier for a decision: offline, the fold model fitted WITHOUT that decision; otherwise the full model."""
    d = json.load(open(OUT / "barrier.json"))
    if decision is not None:
        for m in d["folds"]:
            if decision in m["test"]:
                return Barrier(m)
    return Barrier(d["full"])
