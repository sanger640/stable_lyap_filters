"""Optimiser's curse check (2026-10-17). Optimiser outputs pass the safety rule of the world model they were optimised
against but topple in reality (conf1: 18 of 31 topples after VERIFIED optimiser interventions). Does an independently
trained world model, used only to certify, catch them?

Pool: every offline optimiser output with known real risk (cbfopt_<tag>.json rows, truth_32), tagged by the model it was
optimised against, plus the 21 non-optimised plans per decision (checks/plans.json) as a non-adversarial reference.
For each chunk: exact fork rule (h >= min(h_now, log2 3), o <= O_MAX; 64 futures; h_now from holding still, per model)
under model_OIL, model_CF and the 4 ensemble members ens_v1/ENS0-3.
-> results/jenga/verifier_check.json: real risk of accepted chunks by verifier, for chunks optimised against OIL vs not.
usage: python3 eval/verifier_check.py
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
M = {"OIL": "results/program_b/oil_v1/model/model_OIL.pt", "CF": "results/program_b/cf_v1/model/model_CF.pt",
     **{f"ENS{k}": f"results/program_b/ens_v1/model_ENS{k}.pt" for k in range(4)}}
TAGS = {"b50oil": "OIL", "b50oilp": "OIL", "a0p": "OIL", "b50": "CF", "a30": "CF", "b50cl": "CL", "b50rep": "V3", "cem2": "V3", "al": "V3"}
O_MAX = 0.485


def main():
    import torch
    import cbf_filter as cbff
    import cbf_qp as q
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    cbff.CFG["H_SAFE"] = float(np.log2(3.0)); cbff.CFG["O_MAX"] = O_MAX
    C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
    plans = {r["i"]: np.asarray(r["plans"], np.float32) for r in json.loads((q.OUT / "checks" / "plans.json").read_text())}
    pool = []                                                   # (decision i, 16-step plan, real risk, source, optimised-against)
    for tag, mdl in TAGS.items():
        f = q.OUT / f"cbfopt_{tag}.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rows"]:
            ch = np.asarray(r["chunk"], np.float32); risk = T.get(q._hash(ch))
            if risk is None:
                continue
            p = C["plan16"][r["i"]].copy(); p[1:9] = ch; p[9:, :3] += ch[-1, :3] - C["plan16"][r["i"]][8, :3]
            pool.append((r["i"], p, risk, tag, mdl))
    for i, P in plans.items():
        for p in P[[0, 1, 4, 7, 13, 17]]:                         # policy plan, 3 samples, 2 nudges (non-adversarial reference)
            risk = T.get(q._hash(p[1:9]))
            if risk is not None:
                pool.append((i, p, risk, "plans21", "none"))
    print("pool", len(pool), flush=True)
    mon = cf.Monitor(seed=99); cbf = cbff.CBF(mon); sim = insp._sim()
    verdict = {m: [] for m in M}
    for m, path in M.items():
        mon.model = wm.WorldModel.load(ROOT / path); mon.model.net.eval()
        cache = {}
        for n, (i, p, risk, tag, mdl) in enumerate(pool):
            sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
            z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
            if i not in cache:
                hold = np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32)
                hh, _ = cbf.score(sim, z0, hold[None], 64); cache[i] = float(hh[0])
            h, o = cbf.score(sim, z0, p[None], 64)
            verdict[m].append(bool(h[0] >= min(cache[i], np.log2(3.0)) - 1e-9 and o[0] <= O_MAX))
        print(m, "done", flush=True)
    rows = [{"i": int(i), "risk": float(risk), "tag": tag, "against": mdl, **{m: verdict[m][n] for m in M}} for n, (i, p, risk, tag, mdl) in enumerate(pool)]
    res = {}
    for grp, sel in (("optimised_against_OIL", lambda r: r["against"] == "OIL"), ("optimised_against_other_models", lambda r: r["against"] not in ("OIL", "none")),
                     ("not_optimised_plans21", lambda r: r["against"] == "none")):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        risk = lambda S: (round(float(np.mean([r["risk"] for r in S])), 4), len(S)) if S else (None, 0)
        ens = lambda r, k: sum(r[f"ENS{j}"] for j in range(4)) >= k
        res[grp] = {"all": risk(R), "pass_OIL": risk([r for r in R if r["OIL"]]), "fail_OIL": risk([r for r in R if not r["OIL"]]),
                    "pass_CF": risk([r for r in R if r["CF"]]),
                    "pass_OIL_and_all_4_ENS": risk([r for r in R if r["OIL"] and ens(r, 4)]),
                    "pass_OIL_fail_some_ENS": risk([r for r in R if r["OIL"] and not ens(r, 4)]),
                    "pass_all_4_ENS": risk([r for r in R if ens(r, 4)]), "pass_3of4_ENS": risk([r for r in R if ens(r, 3)])}
    (ROOT / "results/jenga/verifier_check.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
