"""Idea 3 (docs/HJ_REACHABILITY.md): learned viability value V(z) = P(a state-only backup certifies), label-free targets
from our own certificate. usage: python3 eval/hj_value.py targets <train|heldout> ; train ; eval"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt")
os.environ["CBF_CALIB"] = "results/jenga/cbf_calibration_OIL.json"
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
PR = ROOT / "results/program_b/policy_rollouts"
OUT = ROOT / "results/jenga/hj"
ARMS_4B = ["record", "cbfmsoil", "cbfg", "cbfg2", "cbfh2"]
BACKUPS = ["hold", "lift10", "lift20", "back10", "back20"]
N_TRAIN, N_HELD = 3000, 300


def train_states():
    rng = np.random.default_rng(20261023); allk = []
    for arm in ARMS_4B:
        for e in range(14300, 14600):
            f = PR / f"paired_{arm}_s4b/ep{e}.npz"
            if f.exists():
                with np.load(f) as d:
                    allk += [(arm, e, int(k)) for k in d["chunk_index"]]
    return [allk[i] for i in sorted(rng.choice(len(allk), N_TRAIN, replace=False))]


def held_states():
    import hj_trigger as H
    sel = H.select(); rng = np.random.default_rng(20261024)
    return [("record_s4", e, int(k)) for e, k, _ in (sel[i] for i in sorted(rng.choice(len(sel), N_HELD, replace=False)))]


def npz_path(arm, e):
    return PR / (f"paired_{arm}/ep{e}.npz" if arm.endswith("_s4") else f"paired_{arm}_s4b/ep{e}.npz")


def backups(sim):
    t = np.asarray(sim.target, np.float32); site = sim.data.site_xpos[sim.site].copy()
    ramp = (np.arange(16) / 15)[:, None]; base = -site[:2] / max(np.linalg.norm(site[:2]), 1e-6)
    out = []
    for b in BACKUPS:
        p = np.repeat(np.r_[t, 0.0][None], 16, 0).astype(np.float32)
        if b.startswith("lift"):
            p[:, 2] += ramp[:, 0] * int(b[4:]) / 1000
        elif b.startswith("back"):
            p[:, :2] += ramp * base * int(b[4:]) / 1000
        out.append(p)
    return np.stack(out)


def proprio(site, closed):
    return np.r_[site, 1.0 if closed else -1.0].astype(np.float32)


def stage_targets(split):
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    cbff.CFG["H_SAFE"] = float(np.log2(3.0))
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); torch = mon.torch
    states = train_states() if split == "train" else held_states()
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"value_{split}.jsonl"; zf = OUT / f"value_{split}_z"
    zf.mkdir(exist_ok=True)
    done = {(r["arm"], r["ep"], r["k"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
    for n, (arm, e, k) in enumerate(states):
        if (arm, e, k) in done:
            continue
        with np.load(npz_path(arm, e)) as S:
            d = int(np.flatnonzero(S["chunk_index"] == k)[0])
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        mon.rng = np.random.default_rng([20261023, e, k])
        z0n = mon.latent(mon.frame(sim)[None])[0]; z0 = torch.as_tensor(z0n, device="cuda").float()
        h, o = cbf.score(sim, z0, backups(sim), 64)
        ok = (h >= cbff.CFG["H_SAFE"] - cbff.CFG["GAMMA"]) & (o <= cbff.CFG["O_MAX"])
        np.save(zf / f"{arm}_{e}_{k}.npy", np.asarray(z0n, np.float16))
        row = {"arm": arm, "ep": e, "k": k, "target": bool(ok.any()), "ok": [bool(x) for x in ok], "h": [float(x) for x in h],
               "o": [round(float(x), 4) for x in o], "proprio": proprio(sim.data.site_xpos[sim.site], sim.gripper < 50).tolist()}
        with open(f, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if n % 100 == 0:
            print(split, "targets", n, flush=True)
    print("targets done", split, flush=True)


def net():
    import torch.nn as nn

    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(8, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                   nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.m = nn.Sequential(nn.Linear(64 + 4, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, z, p):                                  # z (B, 28, 28, 8), p (B, 4) -> logit (B,)
            import torch
            x = self.c(z.permute(0, 3, 1, 2))
            pp = torch.cat([(p[:, :3] - p.new_tensor([0.5, 0.0, 0.5])) * 10, p[:, 3:]], 1)
            return self.m(torch.cat([x, pp], 1))[:, 0]
    return V()


def load(split):
    R = [json.loads(l) for l in (OUT / f"value_{split}.jsonl").read_text().splitlines()]
    Z = np.stack([np.load(OUT / f"value_{split}_z" / f"{r['arm']}_{r['ep']}_{r['k']}.npy") for r in R]).astype(np.float32)
    P = np.array([r["proprio"] for r in R], np.float32); Y = np.array([r["target"] for r in R], np.float32)
    return R, Z, P, Y


def stage_train():
    import torch
    from wm_libero import auroc
    R, Z, P, Y = load("train")
    eps = sorted({r["ep"] for r in R}); rng = np.random.default_rng(0)
    val_eps = set(rng.choice(eps, len(eps) // 5, replace=False).tolist())
    va = np.array([r["ep"] in val_eps for r in R]); tr = ~va
    torch.manual_seed(0); m = net().cuda(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    Zt, Pt, Yt = (torch.as_tensor(x, device="cuda") for x in (Z, P, Y))
    pos_w = torch.tensor(float((1 - Y[tr]).sum() / max(Y[tr].sum(), 1)), device="cuda")      # balance: weight the rarer class
    best, best_state, log = -1, None, []
    idx = np.flatnonzero(tr)
    for step in range(3000):
        b = torch.as_tensor(rng.choice(idx, 128), device="cuda")
        logit = m(Zt[b], Pt[b]); y = Yt[b]
        w = torch.where(y > 0.5, torch.ones_like(y), pos_w)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 99:
            with torch.no_grad():
                pv = torch.sigmoid(m(Zt[va], Pt[va])).cpu().numpy()
            a = auroc(pv, Y[va] > 0.5); log.append({"step": step + 1, "loss": float(loss), "val_auroc": a})
            if a > best:
                best, best_state = a, {k: v.detach().clone() for k, v in m.state_dict().items()}
    torch.save(best_state, OUT / "value_V.pt")
    out = {"train": int(tr.sum()), "val": int(va.sum()), "target_rate": float(Y.mean()), "best_val_auroc": best, "log": log}
    (OUT / "value_train.json").write_text(json.dumps(out, indent=1)); print({k: v for k, v in out.items() if k != "log"})


def stage_eval():
    """3a on the held-out states; 3b: trigger V3 = V(imagined state after the policy's first chunk) < 0.5 on idea 2's rows."""
    import torch
    from wm_libero import auroc
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    m = net().cuda(); m.load_state_dict(torch.load(OUT / "value_V.pt")); m.eval()
    R, Z, P, Y = load("heldout")
    with torch.no_grad():
        v = torch.sigmoid(m(torch.as_tensor(Z, device="cuda"), torch.as_tensor(P, device="cuda"))).cpu().numpy()
    out = {"3a_heldout": len(R), "3a_target_rate": float(Y.mean()), "3a_auroc": auroc(v, Y > 0.5)}
    out["3a_pass"] = bool(out["3a_auroc"] >= 0.85)
    mon = cf.Monitor(); sim = insp._sim()
    rows = [json.loads(l) for l in (OUT / "trigger_rows.jsonl").read_text().splitlines()]
    f = OUT / "value_v3_rows.jsonl"; done = {(r["ep"], r["k"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
    for r in rows:
        if (r["ep"], r["k"]) in done:
            continue
        with np.load(npz_path("record_s4", r["ep"])) as S:
            d = int(np.flatnonzero(S["chunk_index"] == r["k"])[0])
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))); plan = S["plan16"][d].copy()
        plan[0, :3] = sim.target
        z = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()[None]
        acts = torch.as_tensor(cbff.horizon(plan, cbff.CFG["N_LOOK"]), device="cuda").reshape(1, -1, 8)
        with torch.no_grad():
            for t in range(4):                                          # 4 model steps = the 8 actions of the chunk
                z = mon.model.step(z, acts[:, t]).float()
            vv = float(torch.sigmoid(m(z, torch.as_tensor(proprio(plan[8, :3], plan[8, 3] > 0)[None], device="cuda")))[0])
        with open(f, "a") as fh:
            fh.write(json.dumps({**{k: r[k] for k in ("ep", "k", "kt", "label", "trig_C", "trig_R")}, "V3": vv, "trig_V3": vv < 0.5}) + "\n")
    V3 = [json.loads(l) for l in f.read_text().splitlines()]
    neg = [r for r in V3 if r["kt"] is None]; pos_eps = sorted({r["ep"] for r in V3 if r["kt"] is not None})
    for t in ("trig_C", "trig_R", "trig_V3"):
        rec = []
        for e in pos_eps:
            rs = [r for r in V3 if r["ep"] == e]; kt = rs[0]["kt"]
            rec.append(any(r[t] for r in rs if r["k"] >= kt - 3))
        out[f"3b_{t}"] = {"FPR": float(np.mean([r[t] for r in neg])), "recall": float(np.mean(rec))}
    out["3b_auroc_minusV3_vs_label"] = auroc(-np.array([r["V3"] for r in V3]), np.array([r["label"] for r in V3]))
    (OUT / "value_report.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    {"targets": lambda: stage_targets(sys.argv[2]), "train": stage_train, "eval": stage_eval}[st]()
