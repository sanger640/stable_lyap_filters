"""Idea 2 (docs/HJ_REACHABILITY.md): reachability trigger vs the Stage 4 trigger on saved policy-alone decisions.
usage: python3 eval/hj_trigger.py score ; report"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt")
os.environ["CBF_CALIB"] = "results/jenga/cbf_calibration_OIL.json"
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUN = ROOT / "results/program_b/policy_rollouts/paired_record_s4"
OUT = ROOT / "results/jenga/hj"
EPS, WINDOW, K_AHEAD, N_RANDOM = range(14000, 14300), 8, 3, 600
BACKUPS = ["hold", "back10", "back20", "back30", "lift20"]


def select():
    rng = np.random.default_rng(20261022); pos, neg = [], []
    for e in EPS:
        log = json.loads((RUN / f"ep{e}.json").read_text())["log"]
        kt = next((c["chunk"] for c in log if c.get("topple_started")), None)
        ks = [c["chunk"] for c in log]
        if kt is not None:
            pos += [(e, k, kt) for k in ks if kt - WINDOW <= k <= kt]
        else:
            neg += [(e, k, None) for k in ks]
    neg = [neg[i] for i in sorted(rng.choice(len(neg), N_RANDOM, replace=False))]
    return pos + neg


def candidates(target, plan16):
    p = np.asarray(plan16, np.float32).copy(); p[0, :3] = target
    q8 = p[8].copy(); net = p[8, :3] - p[0, :3]; nrm = np.linalg.norm(net)
    ramp = (np.arange(1, 8) / 7)[:, None]
    out = {}
    for b in BACKUPS:
        tail = np.repeat(q8[None], 7, 0)
        if b.startswith("back"):
            if nrm * 1000 < 2.0:
                continue
            tail[:, :3] = q8[:3] - ramp * (int(b[4:]) / 1000) * net / nrm
        elif b == "lift20":
            tail[:, 2] = q8[2] + ramp[:, 0] * 0.02
        out[b] = np.concatenate([p[:9], tail]).astype(np.float32)
    return out


def stage_score():
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    cbff.CFG["H_SAFE"] = float(np.log2(3.0))
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); torch = mon.torch
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / "trigger_rows.jsonl"
    done = {(r["ep"], r["k"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
    cache = {}
    for n, (e, k, kt) in enumerate(select()):
        if (e, k) in done:
            continue
        if e not in cache:
            cache = {e: dict(np.load(RUN / f"ep{e}.npz"))}
        S = cache[e]; d = int(np.flatnonzero(S["chunk_index"] == k)[0])
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        mon.rng = np.random.default_rng([20261022, e, k])
        z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
        plan = S["plan16"][d]; hold = np.repeat(np.r_[sim.target, 0.0][None].astype(np.float32), 16, 0)
        cands = candidates(sim.target.copy(), plan); names = list(cands)
        h, o = cbf.score(sim, z0, np.stack([plan, hold] + [cands[b] for b in names]), 64)
        ok = cbf.ok(h, o, h[1])
        row = {"ep": e, "k": int(k), "kt": kt, "label": bool(kt is not None and k <= kt <= k + K_AHEAD),
               "trig_C": bool(not ok[0]), "h": [float(x) for x in h], "o": [round(float(x), 4) for x in o],
               "ok_backup": {b: bool(ok[2 + i]) for i, b in enumerate(names)}}
        row["trig_R"] = not any(row["ok_backup"].values())
        row["trig_Rhold"] = not row["ok_backup"]["hold"]
        with open(f, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            print("scored", n, flush=True)
    print("score done", flush=True)


def stage_report():
    R = [json.loads(l) for l in (OUT / "trigger_rows.jsonl").read_text().splitlines()]
    neg = [r for r in R if r["kt"] is None]; pos_eps = sorted({r["ep"] for r in R if r["kt"] is not None})
    out = {"decisions": len(R), "neg_decisions": len(neg), "toppled_episodes": len(pos_eps)}
    for t in ("trig_C", "trig_R", "trig_Rhold"):
        fpr = float(np.mean([r[t] for r in neg]))
        rec, lead = [], []
        for e in pos_eps:
            rs = sorted([r for r in R if r["ep"] == e], key=lambda r: r["k"]); kt = rs[0]["kt"]
            rec.append(any(r[t] for r in rs if r["k"] >= kt - K_AHEAD))
            first = next((r for r in rs if r[t]), None)
            if first is not None:
                lead.append(kt - first["k"])
        tpr = float(np.mean([r[t] for r in R if r["label"]]))
        out[t] = {"FPR": fpr, "recall": float(np.mean(rec)), "median_lead": float(np.median(lead)) if lead else None, "TPR": tpr}
    C, Rr = out["trig_C"], out["trig_R"]
    out["advance_R"] = bool((Rr["recall"] >= C["recall"] - 0.03 and Rr["FPR"] <= C["FPR"])
                            or (Rr["recall"] >= C["recall"] + 0.05 and Rr["FPR"] <= C["FPR"] + 0.02))
    (OUT / "trigger_report.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"score": stage_score, "report": stage_report}[sys.argv[1]]()
