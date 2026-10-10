"""Phase 2 amendment 4: (A) readout candidates for multi-factor scenes, (B) longer / wider dynamics.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_4).
  readout  candidates P, H30, H30_floor, BOTH on H2's real futures (ceiling) and, as the guard, US on the 3-block H
  wide     train US2_wide (width 192, 7,500 steps)        long  train US2_long (22,500 steps)
  grade    US2, US2_wide, US2_long (and the 3-block US zero-shot) on H2 under the selected readout
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_models as pm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase2_generality_protocol.json"
RESULT = ROOT / "results/jenga/phase2_amend4_result.json"
CANDIDATES = ("P", "H30", "H30_floor", "BOTH")


def alarm(e10, e30, rule, tau=0.0):
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    ok = lambda t: bool(t.bic_prefers_two and t.ashman_d > 2 and t.minority >= 2)
    if rule == "P":
        return bool(persistent_mode_alarm(e10, e30).alarm)
    t30 = two_mode_test(e30, 0.0)
    if rule == "H30":
        return ok(t30)
    if rule == "H30_floor":
        return ok(t30) and float(t30.separation_rms_mm) > tau
    return ok(t30) and ok(two_mode_test(e10, 0.0))


def score(a, fork):
    a, fork = np.asarray(a), np.asarray(fork)
    c, f = float((a & fork).sum() / fork.sum()), float((a & ~fork).sum() / (~fork).sum())
    return {"caught": round(c, 4), "false_alarm": round(f, 4), "forks_caught": int((a & fork).sum()),
            "false_alarms": int((a & ~fork).sum()), "score": round(c - f, 4)}


def load_result():
    return json.loads(RESULT.read_text()) if RESULT.exists() else {}


def save(res):
    RESULT.write_text(json.dumps(res, indent=1) + "\n")


def readout():
    import torch
    import phase1_models as p1
    import phase1d_models as p1d
    import step3_world_model as wm
    import step3b_models as sm
    from fresh_fragility_panel import topple_count
    res = load_result()
    # (1) ceiling: real futures on H2 (floor 0 for real futures, so H30_floor = H30 there)
    fork, al = [], {r: [] for r in CANDIDATES}
    for path in sorted((pm.OUT / "h2").glob("*.npz")):
        with np.load(path) as f:
            k = int(f["harm"].sum()); z = f["z"].astype(np.float64).reshape(64, 2, -1)
        fork.append(2 <= k <= 62)
        for r in CANDIDATES:
            al[r].append(alarm(z[:, 0], z[:, 1], r))
    res["A_real_futures_H2"] = {r: score(al[r], fork) for r in CANDIDATES}
    # (2) guard: US on the 3-block H with G starts (as Phase 1e), each candidate
    us = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())["US"]
    model, gop = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), sm.G.load(sm.DATA / "G.pt")
    fork3, al3 = [], {r: [] for r in CANDIDATES}
    with torch.no_grad():
        for path in sorted((p1.OUT / "h").glob("ep*.npz")):
            with np.load(path) as f:
                k = topple_count(f["traces"], f["start_tilt"])
                d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
                z0 = torch.as_tensor(f["z0"], device=sm.DEVICE).float()
                e10, e30 = sm.roll(model, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()),
                                   p1d.augment(f["actions"], None, False))
            fork3.append(2 <= k <= 62)
            for r in CANDIDATES:
                al3[r].append(alarm(e10, e30, r, us["tau"]))
            torch.cuda.empty_cache()
    res["A_guard_US_3block_H"] = {r: score(al3[r], fork3) for r in CANDIDATES}
    base = res["A_guard_US_3block_H"]["P"]["score"]
    qualified = [r for r in CANDIDATES if res["A_guard_US_3block_H"][r]["score"] >= base - 0.03]
    res["A_qualified"] = qualified
    res["A_selected_readout"] = max(qualified, key=lambda r: (res["A_real_futures_H2"][r]["score"], -CANDIDATES.index(r)))
    save(res)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("A_")}, indent=1), flush=True)


def train_arm(arm):
    import step3_world_model as wm
    import fast_train as ft
    import phase1e_models as p1e
    states, names = pm.load_train()
    sampler_te = pm.GaussTE(__import__("torch").load(pm.OUT / "Gauss_TE_phase2.pt"))
    modes = json.loads((pm.OUT / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    if arm == "US2_wide":
        wm.WIDTH = 192
        ft.train(arm, pm.OUT, states, sampler, start_fn=sampler_te)
    elif arm == "US2_long":
        ft.train(arm, pm.OUT, states, sampler, steps=3 * ft.STEPS, start_fn=sampler_te)
    else:                                                   # amendment 5
        ft.train(arm, pm.OUT, states, sampler, steps=9 * ft.STEPS, start_fn=sampler_te)
    print("trained", arm, flush=True)


def grade(key="B_grade_H2"):
    import torch
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    from outcome_modes import two_mode_test
    res = load_result()
    rule = res["A_selected_readout"]
    te = pm.GaussTE(torch.load(pm.OUT / "Gauss_TE_phase2.pt"))
    arms = {a: pm.OUT / f"model_{a}.pt" for a in ("US2", "US2_wide", "US2_long", "US2_xlong")}
    arms["US_3block_zero_shot"] = ROOT / "results/program_b/phase1e/model_US.pt"
    models = {a: wm.WorldModel.load(p) for a, p in arms.items() if p.exists()}
    files = sorted((pm.OUT / "h2").glob("*.npz"))
    fork, fam, ends, errs, spread = [], [], {a: [] for a in models}, {a: [] for a in models}, {a: [] for a in models}
    with torch.no_grad():
        for n_state, path in enumerate(files):
            with np.load(path) as f:
                k = int(f["harm"].sum())
                z0 = torch.as_tensor(f["z0"], device=sm.DEVICE).float()
                g = torch.Generator(device=sm.DEVICE).manual_seed(pm.SEEDS["monitor"] + n_state)
                starts = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
                acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
                real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            fork.append(2 <= k <= 62); fam.append(path.stem.split("_")[0])
            for a, m in models.items():
                e10, e30 = sm.roll(m, starts, acts)
                ends[a].append((e10.astype(np.float32), e30.astype(np.float32)))
                errs[a].extend(np.linalg.norm(e30 - real30, axis=1) / np.sqrt(real30.shape[1] // 3))
                if 2 <= k <= 62:
                    spread[a].append(float(e30.std(0).sum() / real30.std(0).sum()))
            torch.cuda.empty_cache()
    fork, fam = np.array(fork), np.array(fam)
    out = {"readout": rule}
    for a in models:
        tau = float(np.mean(errs[a]))
        al = np.array([alarm(e10.astype(np.float64), e30.astype(np.float64), rule, tau) for e10, e30 in ends[a]])
        out[a] = {"tau": tau, "all": score(al, fork), "F1": score(al[fam == "F1"], fork[fam == "F1"]),
                  "F2": score(al[fam == "F2"], fork[fam == "F2"]), "fork_spread_ratio_median": float(np.median(spread[a]))}
        log = pm.OUT / f"model_{a}_log.json"
        if log.exists():
            steps = json.loads(log.read_text())[0]["steps"]
            out[a]["possibly_undertrained"] = ft.undertrained(json.loads(log.read_text()), steps)
            out[a]["final_loss"] = [r["loss"] for r in json.loads(log.read_text()) if "step" in r][-1]
    trained = [a for a in ("US2", "US2_wide", "US2_long", "US2_xlong") if a in out]
    out["ranking"] = sorted(trained, key=lambda a: out[a]["all"]["score"], reverse=True)
    best = out[out["ranking"][0]]["all"]
    out["best_passes_in_family_bar_on_H2"] = bool(best["caught"] >= 0.80 and best["false_alarm"] <= 0.20)
    res[key] = out
    save(res)
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("readout", "wide", "long", "grade", "xlong", "grade5"))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["amendment_4"]["status"].startswith("frozen"):
        raise SystemExit("amendment 4 not frozen")
    {"readout": readout, "wide": lambda: train_arm("US2_wide"), "long": lambda: train_arm("US2_long"),
     "grade": grade, "xlong": lambda: train_arm("US2_xlong"), "grade5": lambda: grade("amendment5_grade_H2")}[args.stage]()


if __name__ == "__main__":
    main()
