"""Offline check of a readout veto on the final-test filter's selection (75 risky dev decisions).
Candidates = policy plan + 12 policy samples (checks/plans.json) + the filter's 3 primitives. A_rep = closest candidate
satisfying the fork rule (h >= min(h_now, 1), o <= O_MAX), else largest h. A_veto = the same, but candidates whose fixed
readout q exceeds the policy plan's are excluded (if none remain, keep A_rep's choice). Real risk: exact sim, 32 futures."""
import json, os, sys, multiprocessing as mp
import numpy as np
sys.path[:0] = ["eval", "src"]
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
os.environ.setdefault("CBF_WM", os.path.abspath("results/program_b/closedloop_v1/model/model_US.pt"))
import cbf_qp as q


def main():
    import torch, cem_filter as cf, cbf_filter as cbff, inspector_data as insp
    C, K = q.load_cache(); T = json.loads((q.OUT / "truth_32.json").read_text())
    rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
    mon = cf.Monitor(seed=99); cbf = cbff.CBF(mon); sim = insp._sim()
    out, jobs = [], []
    for r in rows:
        i = r["i"]; sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        P = np.asarray(r["plans"], np.float32)[:13]
        cands = np.concatenate([P, np.stack(cbf.primitives(sim, P[0]))]).astype(np.float32)
        h, o = cbf.score(sim, z0, cands, 32); qv = cbf.qfix(sim, z0, cands)
        feas = cbf.ok(h, o, K[i]["h_now"])
        dist = np.array([np.linalg.norm(c[1:9, :3] - P[0][1:9, :3]) for c in cands])
        pick = lambda m: int(np.flatnonzero(m)[np.argmin(dist[m])]) if m.any() else int(np.lexsort((o, -h))[0])
        ja = pick(feas); fv = feas & (qv <= qv[0] + 1e-9); jv = pick(fv) if fv.any() else ja
        out.append({"i": i, "ja": ja, "jv": jv, "q": qv.tolist(), "chunks": [c[1:9].tolist() for c in cands]})
        for j in {ja, jv}:
            ch = cands[j][1:9]; hh = q._hash(ch)
            if hh not in T:
                jobs.append((hh, ((i, 7, 0), C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i], ch))); T[hh] = None
    print(len(jobs), "new truth jobs", flush=True)
    os.environ["TRUTH_N"] = "32"
    with mp.get_context("spawn").Pool(24) as pool:
        for hh, flags in pool.imap_unordered(q._truth_one, jobs):
            T[hh] = sum(flags) / len(flags)
    (q.OUT / "truth_32.json").write_text(json.dumps(T))
    ra = np.array([T[q._hash(np.asarray(o_["chunks"][o_["ja"]], np.float32))] for o_ in out])
    rv = np.array([T[q._hash(np.asarray(o_["chunks"][o_["jv"]], np.float32))] for o_ in out])
    rp = np.array([T[q._hash(np.asarray(o_["chunks"][0], np.float32))] for o_ in out])
    eps = 2 / 32
    res = {"decisions": len(out), "policy": round(float(rp.mean()), 4), "A_rep": round(float(ra.mean()), 4), "A_veto": round(float(rv.mean()), 4),
           "veto_changed_choice": int(sum(o_["ja"] != o_["jv"] for o_ in out)),
           "A_rep_hurt_vs_policy": int((ra > rp + eps).sum()), "A_veto_hurt_vs_policy": int((rv > rp + eps).sum()),
           "A_rep_helped": int((ra < rp - eps).sum()), "A_veto_helped": int((rv < rp - eps).sum())}
    (q.OUT / "veto_check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
