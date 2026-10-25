"""Label-free latent safety filter (results/jenga/lflsf_protocol.json), stage LF1: the margin net l_hat(z).
LSF stage 2 learns its reachability value against a classifier trained on topple LABELS; here the margin is learned from
our label-free hold-plan fork statistics instead (results/jenga/hj2/data): m = neg_log_sens_4 (-log of the 64 futures'
spread per unit perturbation at 4x after the 30-step hold), t = tanh((m - q10) / s) with q10 / s from the training
decisions' own distribution. Same architecture as the LSF classifier, proprio zeroed (state-only). Labels: grading only.
usage: python3 eval/lflsf.py margin   -> results/program_b/lsf_v1/margin_lf.pt, results/jenga/lflsf_LF1.json
       python3 eval/lflsf.py gate2    -> results/jenga/lflsf_LF2.json (after lsf_audit / lsf_diag on value_lf_alea.pt)
       python3 eval/lflsf.py dev      -> results/jenga/lflsf_LF3_dev.json (after outcome_audit of the local dev sweep)
       python3 eval/lflsf.py dev4     -> results/jenga/lflsf_LF4_dev.json (LF3 + LF4 arms, results/jenga/lflsf_LF4_protocol.json)
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
SCALE = float(__import__("os").environ.get("LF_SCALE", "4.0"))         # LF5: 8.0 (results/jenga/lflsf_LF5_protocol.json)
SFX = "" if SCALE == 4.0 else f"{SCALE:g}"


def load(runs):
    Z, M, Y = [], [], []
    for run in runs:
        for f in sorted((DATA / run).glob("ep*.npz")):
            with np.load(f) as d:
                j = list(np.round(d["scales"], 3)).index(SCALE)
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
            best = vl; torch.save({"state": net.state_dict(), "no_g": True, "q10": q10, "s": s, "target": f"neg_log_sens_{SCALE:g}"},
                                  L.OUT / f"margin_lf{SFX}.pt")
    net.load_state_dict(torch.load(L.OUT / f"margin_lf{SFX}.pt")["state"]); net.eval()
    ph = pred(Zh); sel = Yh >= 0
    auc_hat = float(roc_auc_score(Yh[sel] == 1, -ph[sel])); auc_t = float(roc_auc_score(Yh[sel] == 1, -th[sel]))
    rho = float(spearmanr(ph, th).correlation)
    out = {"train_decisions": int(len(tt)), "held_decisions": int(len(th)), "q10": q10, "s": s,
           "held_spearman": round(rho, 3), "held_auc_hat": round(auc_hat, 3), "held_auc_target": round(auc_t, 3),
           "held_share_hat_below0": round(float((ph < 0).mean()), 3), "held_share_target_below0": round(float((th < 0).mean()), 3),
           "best_val_mse": best, "log": log}
    out["GATE_LF1_pass"] = bool(rho >= 0.6 and auc_hat >= auc_t - 0.05)
    (ROOT / f"results/jenga/lflsf_LF1{'_x' + SFX if SFX else ''}.json").write_text(json.dumps(out, indent=1)); print({k: v for k, v in out.items() if k != "log"})


def gate2():
    stem = __import__("os").environ.get("LSF_VALUE_NAME", "value_lf_alea.pt")[:-3]
    a = json.loads((ROOT / f"results/jenga/lsf_audit_{stem}.json").read_text())
    d = json.loads((ROOT / f"results/jenga/lsf_value_diag_{stem}.json").read_text())
    out = {"actor_saturated_share": a["actor"]["actor_saturated_share"], "V_after_actor": a["actor"]["V_after_plan_median"],
           "V_after_policy": a["policy"]["V_after_plan_median"], "V_after_hold": a["hold"]["V_after_plan_median"],
           "AUROC_minusV_vs_any_topple": d["AUROC_minusV_vs_any_topple"], "AUROC_minus_l_now": d["AUROC_minus_l_now_vs_any_topple"],
           "V_quantiles": d["V_quantiles"], "share_V_below_0.3": d["share_V_below_0.3"]}
    out["GATE_LF2_pass"] = bool(out["actor_saturated_share"] < 0.05 and out["V_after_actor"] >= out["V_after_hold"]
                                and out["AUROC_minusV_vs_any_topple"] >= 0.70)
    (ROOT / f"results/jenga/lflsf_LF2{'_x' + SFX if SFX else ''}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


DEV_ARMS = ("lfa-25", "lfa-15", "lfa000", "lfa010", "lfa030", "lfa050", "lfr010", "lfr030", "lfr050")   # amendments 1, 2


def dev():
    """LF3 selection (the LSF dev rule): fewest replay-audited topples among arms with v4 >= policy - 5 pp; ties -> larger eps."""
    A = json.loads((ROOT / "results/jenga/outcome_audit_lflsfdev.json").read_text())
    R = {a: {r["ep"]: r for r in A[f"paired_{a}_lflsfdev"]["rows"]} for a in ("record",) + DEV_ARMS}
    EPS = sorted(set.intersection(*(set(v) for v in R.values())))
    keys = ("logged_neighbour_toppled", "v4_success", "v4b_success", "red_fell_before_extraction", "red_fell_after_extraction")
    cnt = {a: {k: sum(bool(R[a][e][k]) for e in EPS) for k in keys} for a in R}
    for a in R:
        cnt[a]["timeout"] = sum(not R[a][e]["extracted"] and not R[a][e]["logged_neighbour_toppled"] and not R[a][e]["red_fell_before_extraction"] for e in EPS)
    pv4 = cnt["record"]["v4_success"] / len(EPS)
    ok = [a for a in DEV_ARMS if cnt[a]["v4_success"] / len(EPS) >= pv4 - 0.05]
    sel = min(ok, key=lambda a: (cnt[a]["logged_neighbour_toppled"], -int(a[3:]))) if ok else None
    iv = {}
    for a in DEV_ARMS:
        L = [json.loads((ROOT / f"results/program_b/policy_rollouts/paired_{a}_lflsfdev/ep{e}.json").read_text())["log"] for e in EPS]
        iv[a] = {"overrides_per_episode": round(float(np.mean([sum(c.get("lsf_overrides", 0) for c in l) for l in L])), 2),
                 "model_steps_per_episode": round(float(np.mean([sum(c.get("lsf_steps", 0) for c in l) for l in L])), 1),
                 "rel_kept_per_episode": round(float(np.mean([sum(c.get("lsf_rel_kept", 0) for c in l) for l in L])), 2)}
    out = {"episodes": len(EPS), "counts": cnt, "eligible_v4_within_5pp": ok, "selected": sel, "interventions": iv,
           "replay_flag_matches": {a: A[f"paired_{a}_lflsfdev"].get("replay_topple_flag_matches") for a in R}}
    (ROOT / "results/jenga/lflsf_LF3_dev.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def dev4():
    """LF4 (results/jenga/lflsf_LF4_protocol.json): the LSF dev rule over all LF3 + LF4 arms; ties -> fewer overrides."""
    A = json.loads((ROOT / "results/jenga/outcome_audit_lflsfdev.json").read_text())
    A.update(json.loads((ROOT / "results/jenga/outcome_audit_lflsfdev4.json").read_text()))
    arms = [k[len("paired_"):-len("_lflsfdev")] for k in A if k != "paired_record_lflsfdev" and k.endswith("_lflsfdev")]
    R = {a: {r["ep"]: r for r in A[f"paired_{a}_lflsfdev"]["rows"]} for a in ["record"] + arms}
    EPS = sorted(set.intersection(*(set(v) for v in R.values())))
    keys = ("logged_neighbour_toppled", "v4_success", "v4b_success", "red_fell_before_extraction", "red_fell_after_extraction")
    cnt = {a: {k: sum(bool(R[a][e][k]) for e in EPS) for k in keys} for a in R}
    iv = {}
    for a in R:
        cnt[a]["timeout"] = sum(not R[a][e]["extracted"] and not R[a][e]["logged_neighbour_toppled"] and not R[a][e]["red_fell_before_extraction"] for e in EPS)
        if a == "record":
            continue
        L = [json.loads((ROOT / f"results/program_b/policy_rollouts/paired_{a}_lflsfdev/ep{e}.json").read_text())["log"] for e in EPS]
        ov = sum(c.get("lsf_overrides", 0) for l in L for c in l)
        iv[a] = {"overrides_per_episode": round(ov / len(EPS), 2),
                 "edit_mm_per_override": round(sum(c.get("lsf_edit_mm", 0.0) for l in L for c in l) / max(ov, 1), 2),
                 "sec_per_model_step": round(sum(c.get("seconds", 0.0) for l in L for c in l) / max(1, sum(c.get("lsf_steps", 0) for l in L for c in l)), 3)}
    pv4 = cnt["record"]["v4_success"] / len(EPS)
    ok = [a for a in arms if cnt[a]["v4_success"] / len(EPS) >= pv4 - 0.05]
    sel = min(ok, key=lambda a: (cnt[a]["logged_neighbour_toppled"], iv[a]["overrides_per_episode"])) if ok else None
    out = {"episodes": len(EPS), "counts": cnt, "interventions": iv, "eligible_v4_within_5pp": ok, "selected": sel}
    (ROOT / "results/jenga/lflsf_LF4_dev.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"margin": margin, "gate2": gate2, "dev": dev, "dev4": dev4}[sys.argv[1]]()
