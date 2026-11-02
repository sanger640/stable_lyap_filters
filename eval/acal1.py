"""ACAL1 (results/jenga/acal1_protocol.json): one barrier strength for every value. The CBF check
min_j V(z'_j) > (1 - gamma) V(z) means a different strictness for every value, because values differ in how much V drops on
an ordinary move. Normalisation (label-free): give each value the gamma at which its check fails on the same share of the
policy's own moves as ours90's check (value_irr8_alea.pt, gamma 0.1) does.
Calibration moves: RNG1's 3000 seeded decisions of paired_record_ocl (policy alone; eval/rng1.py calib_select). Each is the
filter's first check of the chunk, exactly as eval/cbf_filter.py lsf_edit: z = the stored fixed-camera latent, g = (target,
first gripper command), the policy's commands plan16[1:3], 8 perceived starts + execution offsets (lsf.Noise at the
scale), one model step, V(z'_j, g'_j). The 8 next latents are shared by all values.
usage: python3 eval/acal1.py run -> results/jenga/acal1_rows.npz;  grade -> results/jenga/acal1_result.json (prints FX_G2 / FX_G1);
       dev -> gate D on the dev arms (prints DEV_PASS);  test -> results/jenga/acal1_test.json"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
os.environ.setdefault("LSF_ALEA", "8")
OUT = ROOT / "results/jenga"
REF = "value_irr8_alea.pt"
VALUES = {"ours": REF, "fx": "value_nsif16_alea.pt", "fq": "value_nsiq16_alea.pt", "nm": "value_irrnm8_alea.pt",
          "seed2": "value_irr8s2_alea.pt", "irr16": "value_irr16_alea.pt"}
SCALES, G_REF, SEED = (2.0, 1.0), 0.1, 20261102
GRID = np.round(np.arange(0.01, 0.6001, 0.001), 3)


def run():
    import torch
    import lsf as L
    import rng1
    import step3_world_model as wm
    lim = int(os.environ.get("ACAL1_LIMIT", 10**9))
    mdl = wm.WorldModel.load(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); mdl.net.eval()
    vfs = {k: L.ValueFn(v) for k, v in VALUES.items()}
    noise = {}
    for s in SCALES:
        L.ALEA_SCALE = s; noise[s] = L.Noise()
    sel = rng1.calib_select()[:lim]; n = 8
    res = {k: [] for k in ("ep", "i", "closed")}; V = {f"{k}_{t}_{s:g}": [] for k in VALUES for t in ("now", "min") for s in SCALES}
    cache = {}
    for c, (run_, stem, i, _, k, cl) in enumerate(sel):
        if stem not in cache:
            cache.clear()
            with np.load(ROOT / "results/jenga/hj2/data" / run_ / f"{stem}.npz") as d:
                hz = d["z"].astype(np.float32)
            with np.load(ROOT / "results/program_b/policy_rollouts" / run_ / f"{stem}.npz") as d:
                plan16, target = d["plan16"], d["target"]
            cache[stem] = (hz, plan16, target)
        hz, plan16, target = cache[stem]
        cmds = np.asarray(plan16[i][1:3], np.float32)                          # PLAN_START = 1: the chunk's first model step
        z = torch.as_tensor(hz[i], device="cuda")[None]
        g = torch.as_tensor(np.r_[target[i], cmds[0][3]][None], dtype=torch.float32, device="cuda")
        a = torch.as_tensor(cmds.reshape(1, 8), device="cuda")
        G2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1).repeat_interleave(n, 0)
        with torch.no_grad():
            for s in SCALES:
                nz = noise[s]; nz.rng = np.random.default_rng([SEED, int(stem[2:]), i, int(s * 10)])
                zs = nz.starts(z.expand(n, *z.shape[1:]).contiguous())
                off = nz.actions(a, n)[:, 0] - a
                act = mdl._act(a + off)
                r = mdl.net(zs, torch.zeros_like(zs), act, torch.zeros(n, device="cuda"))
                zn = (zs + r * mdl.stats.res_std).float()
                for key, vf in vfs.items():
                    V[f"{key}_now_{s:g}"].append(float(vf.V(z, g))); V[f"{key}_min_{s:g}"].append(float(vf.V(zn, G2).min()))
        for key, v in (("ep", int(stem[2:])), ("i", i), ("closed", cl)):
            res[key].append(v)
        if c % 500 == 0:
            print("acal1", c, len(sel), flush=True)
    np.savez_compressed(OUT / "acal1_rows.npz", **{k: np.asarray(v) for k, v in res.items()},
                        **{k: np.asarray(v, np.float32) for k, v in V.items()})
    print("saved", flush=True)


def fail_share(vnow, vmin, gam):
    return float((vmin <= (1 - gam) * vnow).mean())                            # lsf_edit: edit iff v0 <= eps


def grade():
    D = np.load(OUT / "acal1_rows.npz"); out = {"protocol": "results/jenga/acal1_protocol.json", "decisions": int(len(D["ep"])), "scales": {}}
    for s in SCALES:
        tgt = fail_share(D[f"ours_now_{s:g}"], D[f"ours_min_{s:g}"], G_REF); o = {"target_fail_share": round(tgt, 4), "values": {}}
        for key in VALUES:
            vn, vm = D[f"{key}_now_{s:g}"], D[f"{key}_min_{s:g}"]
            F = np.array([fail_share(vn, vm, g_) for g_ in GRID])
            d = np.abs(F - tgt); gam = round(float(np.median(GRID[d == d.min()])), 3)   # closest share; ties -> middle of the tie
            rel = (vn - vm) / np.maximum(np.abs(vn), 1e-6)
            o["values"][key] = {"file": VALUES[key], "gamma": gam, "fail_share_at_0.1": round(fail_share(vn, vm, 0.1), 4),
                                "fail_share_at_0.2": round(fail_share(vn, vm, 0.2), 4),
                                "fail_share_at_gamma": round(fail_share(vn, vm, gam), 4),
                                "median_V_now": round(float(np.median(vn)), 4), "share_V_now_below0": round(float((vn < 0).mean()), 4),
                                "median_rel_drop": round(float(np.median(rel[vn > 0])), 4),
                                "q90_rel_drop": round(float(np.quantile(rel[vn > 0], 0.9)), 4)}
        out["scales"][f"{s:g}x"] = o
    if out["scales"]["1x"]["target_fail_share"] * out["decisions"] < 30:     # too few failing moves at 1x to match: use 2x's gamma
        for key in VALUES:
            out["scales"]["1x"]["values"][key]["gamma_matched"] = out["scales"]["1x"]["values"][key]["gamma"]
            out["scales"]["1x"]["values"][key]["gamma"] = out["scales"]["2x"]["values"][key]["gamma"]
        out["scales"]["1x"]["fallback_to_2x"] = True
    v = out["scales"]["2x"]["values"]
    ratio = v["fx"]["fail_share_at_0.1"] / max(v["ours"]["fail_share_at_0.1"], 1e-9)
    out["gate_N"] = {"rule": "fx's check at gamma 0.1 fails on >= 1.5x as many of the policy's moves as ours90's (2x)",
                     "ratio": round(ratio, 3), "pass": bool(ratio >= 1.5)}
    out["fx_gamma"] = {f"{s:g}x": out["scales"][f"{s:g}x"]["values"]["fx"]["gamma"] for s in SCALES}
    out["ours_gamma_check"] = {f"{s:g}x": out["scales"][f"{s:g}x"]["values"]["ours"]["gamma"] for s in SCALES}
    (OUT / "acal1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    print(f"GATE_N={int(out['gate_N']['pass'])}"); print(f"FX_G2={out['fx_gamma']['2x']}"); print(f"FX_G1={out['fx_gamma']['1x']}")


def dev():
    import irrm_select as S
    c2, c1 = S.counts("ac1fxndev2x"), S.counts("ac1fxndev1x")
    ok = bool(c2["v4"] >= 65 and c2["timeouts"] <= 20 and c1["v4"] >= 80 and c1["topples"] <= 5)
    out = json.loads((OUT / "acal1_result.json").read_text()); out["gate_D"] = {"2x": c2, "1x": c1, "pass": ok,
        "rule": "2x v4 >= 65 and time-outs <= 20; 1x v4 >= 80 and topples <= 5 (dev 15000-15099)"}
    (OUT / "acal1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["gate_D"])); print(f"DEV_PASS={int(ok)}")


def test():
    import h2h5_analysis as H
    H.ARMS = {"ours90": "paired_lfc000_ac1o90_{n}", "fxn": "paired_lfc000_ac1fxn_{n}"}
    res, P = {}, {}
    for nz in ("2x", "1x"):
        A = {}
        for t_ in ("o90", "fxn"):
            A.update(json.loads((OUT / f"outcome_audit_ac1{t_}_{nz}.json").read_text()))
        V = {a: {r["ep"]: r for r in A[run.format(n=nz)]["rows"]} for a, run in H.ARMS.items()}
        eps = sorted(set.intersection(*(set(v) for v in V.values())))
        r = {"episodes": len(eps), "counts": {a: {k: int(H.vec(V, eps, a, k).sum()) for k in H.KEYS} for a in H.ARMS},
             "edits_per_episode": {a: round(float(np.mean([sum(c.get("lsf_overrides", 0) or 0 for c in json.loads(
                 (H.RUNS / H.ARMS[a].format(n=nz) / f"ep{e}.json").read_text())["log"]) for e in eps])), 2) for a in H.ARMS}}
        r["pairs"] = {"fxn_vs_ours90": {k: H.paired(V, eps, "fxn", "ours90", k) for k in H.KEYS}}
        res[nz] = r; P[nz] = r["pairs"]["fxn_vs_ours90"]
    noninf = lambda nz: bool(P[nz]["topple"]["upper95_pp"] < 3 and P[nz]["v4_success"]["lower95_pp"] > -5)
    hyp = {"T_N_2x": noninf("2x"), "T_N_1x": noninf("1x")}
    hyp["T_N_better"] = bool(hyp["T_N_2x"] and (P["2x"]["topple"]["upper95_pp"] < 0 or P["2x"]["v4_success"]["lower95_pp"] > 0))
    hyp["switch_candidate"] = bool(hyp["T_N_2x"] and hyp["T_N_1x"] and hyp["T_N_better"])
    res["hypotheses"] = hyp
    (OUT / "acal1_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"hypotheses": hyp, **{nz: {k: res[nz][k] for k in ("counts", "edits_per_episode")} for nz in ("2x", "1x")},
                      "pairs": P}, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade, "dev": dev, "test": test}[sys.argv[1]]()
