"""IRRM stage 2 (results/jenga/irrm_protocol.json, amendment): the learned irreversibility margin l_irr(z, g, d).
Labels: the undo test of eval/irrm1.py at horizon NL control steps (IRRM_NL, stage 1's selection) for an ACTION of the
value's actor (normalised target motion d in [-1, 1]^6, 2 control steps x xyz x lsf.MAX_STEP, gripper held; for NL = 8 the
same 2-step motion repeated over the move): do it, undo back to the current target, hold; vs hold; 64 draws at 2x noise
(perception through the start operator G, execution snippets), r_max = max_j q98_outside-hand |f_rev_j - f_hold_j|.
  states  (train): the value's reset states (lsf_v1/train_reset_*.npy, 5969) and, for each, one imagined state 1-8 model
                   steps later under random actor motions (the value trains mostly on imagined latents)
  actions (train): per state the policy's own first 2 steps (clipped to the actor range; resets only), d = 0 (hold) and
                   3 uniform random d
  held-out:        1500 decisions of paired_record_m3 (not a value-training run): the policy's action + 1 random
Margin target t = tanh((THR - r_max) / s), THR = 1.3356, s = IQR(r_max over the train labels) / 2 (label-free);
net = lsf Critic architecture (trunk + MLP on z, g, [d, 0, 0]) with tanh.
usage: SHARD=i/n python3 eval/irrm2.py labels;  python3 eval/irrm2.py held;  python3 eval/irrm2.py margin
       -> results/jenga/irrm2_labels_nl<NL>_<i>.npz, irrm2_held_nl<NL>.npz, lsf_v1/margin_irr_nl<NL>.pt, results/jenga/irrm2_result_nl<NL>.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga"
S, THR, N_FUT, SEED = 2.0, 1.3356, 64, 20261027
NL = int(os.environ.get("IRRM_NL", 2))
K_RAND = 3


class Undo:
    """the undo test for a batch of actions from one latent (shared hold branch: same draws)"""
    def __init__(self):
        import torch
        import cbf_barrier as cb
        import cbf_filter as cbff
        import cem_filter as cf
        import step3b_models as sm
        import step3_world_model as wm
        import visual_pipeline_panel3 as p3
        import uncertainty_monitor_dev as um
        import lsf as L
        self.torch, self.cb, self.wm, self.L = torch, cb, wm, L
        self.model = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); self.model.net.eval()
        self.G = sm.G.load(cf.V3 / "G.pt"); self.E = p3.ErrorModel(); self.snip = np.load(um.BENCH)["snippets"]
        self.total = 2 * wm.STEPS

    def targets(self, g, d):
        """g (4,) current target + gripper, d (6,) normalised -> (NL, 3) absolute targets (2-step motion, repeated to NL)"""
        st = np.asarray(d, np.float32).reshape(2, 3) * self.L.MAX_STEP
        steps = np.concatenate([st] * (NL // 2))
        return g[:3] + np.cumsum(steps, 0)

    def r_max(self, z0, g, D, seed):
        torch, wm = self.torch, self.wm
        z0 = torch.as_tensor(z0, device="cuda").float()
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                       for p, r in (self.E.sample(seed, 0, 0, j) for j in range(N_FUT))])
        st = self.G(z0[None].expand(N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        cur = np.r_[g[:3], g[3]].astype(np.float32)
        noise = [np.concatenate([self.snip[j % 64], self.snip[(j + 1) % 64]])[:NL] * S for j in range(N_FUT)]
        bases, masks = [], []
        for d in D:
            seg = self.targets(g, d)
            path = np.concatenate([seg, seg[::-1][1:], g[None, :3]])                 # do, then the same targets back to the current one
            rev = np.c_[path, np.full(len(path), g[3])].astype(np.float32)
            bases.append(np.concatenate([rev, np.repeat(cur[None], self.total, 0)])[:self.total])
            masks.append(torch.as_tensor(~self.cb.hand_mask(np.vstack([g[None, :3], seg])), device="cuda"))
        bases.append(np.repeat(cur[None], self.total, 0))
        acts = []
        for base in bases:
            for j in range(N_FUT):
                a = base.copy(); a[:NL, :3] -= noise[j]; acts.append(a)
        B = len(bases)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(B * N_FUT, wm.STEPS, -1)
        z = st.repeat(B, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = self.model.step(z, pairs[:, t]).float()
        z = z.float().reshape(B, N_FUT, *z.shape[1:])
        return np.array([float(torch.quantile((z[b] - z[-1]).abs().mean(-1)[:, masks[b]], 0.98, dim=1).max()) for b in range(len(D))])

    def imagine(self, z0, g, rng, steps):
        """roll the deterministic model `steps` model steps under random actor motions -> (z, g)"""
        torch = self.torch
        z = torch.as_tensor(z0, device="cuda").float()[None]; gg = torch.as_tensor(g, device="cuda").float()[None]
        with torch.no_grad():
            for _ in range(steps):
                d = torch.as_tensor(rng.uniform(-1, 1, (1, 6)), device="cuda").float()
                a = self.L.plan_from_d(gg, d); z = self.model.step(z, a).float(); gg = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
        return z[0].cpu().numpy(), gg[0].cpu().numpy()


def labels():
    i_sh, n_sh = map(int, os.environ.get("SHARD", "0/1").split("/"))
    U = Undo(); L = U.L
    Rz = np.load(L.OUT / "train_reset_z.npy"); Rg = np.load(L.OUT / "train_reset_prop.npy"); Rp = np.load(L.OUT / "train_reset_plan.npy")
    out = {k: [] for k in ("z", "g", "d", "r", "src")}
    for n, i in enumerate(range(i_sh, len(Rz), n_sh)):
        rng = np.random.default_rng([SEED, i])
        g = Rg[i].astype(np.float32)
        p = Rp[i, :2, :3]; dpol = np.clip(np.concatenate([p[0] - g[:3], p[1] - p[0]]) / L.MAX_STEP, -1, 1)
        for src in (0, 1):                                                        # 0 reset state, 1 imagined state
            if src == 0:
                z, gg = Rz[i].astype(np.float32), g
                D = [dpol, np.zeros(6)] + list(rng.uniform(-1, 1, (K_RAND, 6)))
            else:
                z, gg = U.imagine(Rz[i], g, rng, int(rng.integers(1, 9)))
                D = [np.zeros(6)] + list(rng.uniform(-1, 1, (K_RAND + 1, 6)))
            r = U.r_max(z, gg, D, SEED + 7919 * i + src)
            for d, rr in zip(D, r):
                out["z"].append(z.astype(np.float16)); out["g"].append(gg); out["d"].append(np.asarray(d, np.float32)); out["r"].append(rr); out["src"].append(src)
        if n % 200 == 0:
            print("irrm2 labels", i_sh, n, flush=True)
    np.savez_compressed(OUT / f"irrm2_labels_nl{NL}_{i_sh}.npz", **{k: np.asarray(v) for k, v in out.items()})
    print("saved", flush=True)


def held():
    """1500 paired_record_m3 decisions: the policy's first 2 steps (clipped) + 1 random action; topple grading labels"""
    U = Undo(); L = U.L
    data = ROOT / "results/jenga/hj2/data/paired_record_m3"; runs = ROOT / "results/program_b/policy_rollouts/paired_record_m3"
    rows = []
    for f in sorted(data.glob("ep*.npz")):
        with np.load(f) as d:
            kt, k = int(d["topple_chunk"]), d["chunk_index"]
        rows += [(f.stem, i, kt, int(k[i])) for i in range(len(k))]
    rng = np.random.default_rng(SEED); sel = [rows[j] for j in sorted(rng.choice(len(rows), 1500, replace=False))]
    out = {k: [] for k in ("z", "g", "d", "r", "pol", "kt", "k")}
    for n, (stem, i, kt, k) in enumerate(sel):
        with np.load(data / f"{stem}.npz") as d:
            z, pr = d["z"][i].astype(np.float32), d["proprio"][i]
        with np.load(runs / f"{stem}.npz") as d:
            plan, target = d["plan16"][i], d["target"][i]
        g = np.r_[target, plan[0, 3]].astype(np.float32)
        p = plan[1:3, :3]; dpol = np.clip(np.concatenate([p[0] - g[:3], p[1] - p[0]]) / L.MAX_STEP, -1, 1)
        D = [dpol, rng.uniform(-1, 1, 6)]
        r = U.r_max(z, g, D, SEED + 104729 * int(stem[2:]) + i)
        for j, (dd, rr) in enumerate(zip(D, r)):
            for key, v in (("z", z.astype(np.float16)), ("g", g), ("d", np.asarray(dd, np.float32)), ("r", rr), ("pol", j == 0), ("kt", kt), ("k", k)):
                out[key].append(v)
        if n % 200 == 0:
            print("irrm2 held", n, flush=True)
    np.savez_compressed(OUT / f"irrm2_held_nl{NL}.npz", **{k: np.asarray(v) for k, v in out.items()})
    print("saved", flush=True)


def margin():
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    import lsf as L
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    fs = sorted(OUT.glob(f"irrm2_labels_nl{NL}_*.npz"))
    T = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("z", "g", "d", "r", "src")}
    H = dict(np.load(OUT / f"irrm2_held_nl{NL}.npz"))
    s = float(np.subtract(*np.quantile(T["r"], [0.75, 0.25]))) / 2
    tgt = lambda r: np.tanh((THR - r) / s).astype(np.float32)
    tt, th = tgt(T["r"]), tgt(H["r"])
    _, Critic, _ = L.nets(); net = Critic().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    a8 = lambda d: torch.cat([d, torch.zeros(len(d), 2, device="cuda")], 1)
    f = lambda z, g, d: torch.tanh(net(z, g, a8(d)))

    def pred(Z, Gg, Dd):
        out = []
        with torch.no_grad():
            for b in range(0, len(Z), 2048):
                out.append(f(*(torch.as_tensor(x[b:b + 2048], device="cuda").float() for x in (Z, Gg, Dd))).cpu().numpy())
        return np.concatenate(out)
    va = rng.random(len(tt)) < 0.05; idx = np.flatnonzero(~va); best, log = 1e9, []
    w = np.where(T["r"][idx] > THR, 4.0, 1.0); w /= w.sum()                        # irreversible actions are rare: oversample 4x
    for ep in range(40):
        net.train()
        for _ in range(len(idx) // 512):
            i = np.sort(rng.choice(idx, 512, p=w))
            z, g, d = (torch.as_tensor(x[i], device="cuda").float() for x in (T["z"], T["g"], T["d"]))
            loss = ((f(z, g, d) - torch.as_tensor(tt[i], device="cuda")) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); vl = float(np.mean((pred(T["z"][va], T["g"][va], T["d"][va]) - tt[va]) ** 2)); log.append({"epoch": ep, "val_mse": vl}); print(log[-1], flush=True)
        if vl < best:
            best = vl; torch.save({"state": net.state_dict(), "thr": THR, "s": s, "nl": NL, "kind": "irr_margin_zgd"}, L.OUT / f"margin_irr_nl{NL}.pt")
    net.load_state_dict(torch.load(L.OUT / f"margin_irr_nl{NL}.pt")["state"]); net.eval()
    ph = pred(H["z"], H["g"], H["d"]); fail = H["r"] > THR
    pol = H["pol"].astype(bool); kt, k = H["kt"], H["k"]; pre = pol & (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = pol & (kt < 0)
    out = {"nl": NL, "train_labels": int(len(tt)), "train_share_irreversible": round(float((T["r"] > THR).mean()), 4), "s": s,
           "held_labels": int(len(th)), "held_share_irreversible": round(float(fail.mean()), 4),
           "held_spearman": round(float(spearmanr(ph, th).correlation), 3),
           "held_auc_hat_vs_true_flag": round(float(roc_auc_score(fail, -ph)), 3) if 0 < fail.sum() < len(fail) else None,
           "held_hat_below0_given_true_flag": round(float((ph[fail] < 0).mean()), 3) if fail.any() else None,
           "held_hat_below0_given_true_safe": round(float((ph[~fail] < 0).mean()), 4),
           "grading_policy_action_auc_pre_vs_safe": {"hat": round(float(roc_auc_score(pre[pre | safe], -ph[pre | safe])), 3),
                                                     "true_r": round(float(roc_auc_score(pre[pre | safe], H["r"][pre | safe])), 3)},
           "best_val_mse": best, "log": log}
    (OUT / f"irrm2_result_nl{NL}.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    {"labels": labels, "held": held, "margin": margin}[sys.argv[1]]()
