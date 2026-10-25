"""Label-free latent safety filter (results/jenga/lflsf_protocol.json), stage LF1: the margin net l_hat(z).
LSF stage 2 learns its reachability value against a classifier trained on topple LABELS; here the margin is learned from
our label-free hold-plan fork statistics instead (results/jenga/hj2/data): m = neg_log_sens_4 (-log of the 64 futures'
spread per unit perturbation at 4x after the 30-step hold), t = tanh((m - q10) / s) with q10 / s from the training
decisions' own distribution. Same architecture as the LSF classifier, proprio zeroed (state-only). Labels: grading only.
usage: python3 eval/lflsf.py margin   -> results/program_b/lsf_v1/margin_lf.pt, results/jenga/lflsf_LF1.json
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
DATA = ROOT / "results/jenga/hj2/data"
TRAIN = ("paired_record_s4", "paired_record_h2h2", "paired_record_ocl", "paired_cbfg_s4", "paired_cbfg_h2h2")
HELD = ("paired_record_m3",)
SEED = 20261025


def load(runs):
    Z, M, Y = [], [], []
    for run in runs:
        for f in sorted((DATA / run).glob("ep*.npz")):
            with np.load(f) as d:
                j = list(np.round(d["scales"], 3)).index(4.0)
                sp = d["stats"][:, j, 7]                                        # spread30 at 4x
                kt, k = int(d["topple_chunk"]), d["chunk_index"]
                Z.append(d["z"]); M.append(-np.log(np.maximum(sp, 1e-12) / 16.0))
                Y.append(np.where(kt < 0, 0, np.where((kt - k >= 1) & (kt - k <= 3), 1, -1)))   # grading: 0 safe ep, 1 pre, -1 other
    return np.concatenate(Z), np.concatenate(M).astype(np.float32), np.concatenate(Y)


def margin():
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    import lsf as L
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Zt, Mt, _ = load(TRAIN); Zh, Mh, Yh = load(HELD)
    q10 = float(np.quantile(Mt, 0.10)); s = float(np.subtract(*np.quantile(Mt, [0.75, 0.25]))) / 2
    T = lambda m: np.tanh((m - q10) / s).astype(np.float32)
    tt, th = T(Mt), T(Mh)
    va = rng.random(len(tt)) < 0.05                                             # early-stopping split of the training runs
    Clf, _, _ = L.nets(); net = Clf().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    g0 = torch.zeros(1, 4, device="cuda")

    def pred(Z):
        out = []
        with torch.no_grad():
            for b in range(0, len(Z), 2048):
                z = torch.as_tensor(Z[b:b + 2048], device="cuda").float(); out.append(net(z, g0.expand(len(z), -1)).cpu().numpy())
        return np.concatenate(out)
    best, log, idx_tr = 1e9, [], np.flatnonzero(~va)
    for ep in range(30):
        net.train()
        for b in range(0, len(idx_tr), 512):
            i = np.sort(idx_tr[rng.integers(len(idx_tr), size=512)])
            z = torch.as_tensor(Zt[i], device="cuda").float(); y = torch.as_tensor(tt[i], device="cuda")
            loss = ((net(z, g0.expand(len(z), -1)) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); vl = float(np.mean((pred(Zt[va]) - tt[va]) ** 2)); log.append({"epoch": ep, "val_mse": vl}); print(log[-1], flush=True)
        if vl < best:
            best = vl; torch.save({"state": net.state_dict(), "no_g": True, "q10": q10, "s": s, "target": "neg_log_sens_4"},
                                  L.OUT / "margin_lf.pt")
    net.load_state_dict(torch.load(L.OUT / "margin_lf.pt")["state"]); net.eval()
    ph = pred(Zh); sel = Yh >= 0
    auc_hat = float(roc_auc_score(Yh[sel] == 1, -ph[sel])); auc_t = float(roc_auc_score(Yh[sel] == 1, -th[sel]))
    rho = float(spearmanr(ph, th).correlation)
    out = {"train_decisions": int(len(tt)), "held_decisions": int(len(th)), "q10": q10, "s": s,
           "held_spearman": round(rho, 3), "held_auc_hat": round(auc_hat, 3), "held_auc_target": round(auc_t, 3),
           "held_share_hat_below0": round(float((ph < 0).mean()), 3), "held_share_target_below0": round(float((th < 0).mean()), 3),
           "best_val_mse": best, "log": log}
    out["GATE_LF1_pass"] = bool(rho >= 0.6 and auc_hat >= auc_t - 0.05)
    (ROOT / "results/jenga/lflsf_LF1.json").write_text(json.dumps(out, indent=1)); print({k: v for k, v in out.items() if k != "log"})


if __name__ == "__main__":
    {"margin": margin}[sys.argv[1]]()
