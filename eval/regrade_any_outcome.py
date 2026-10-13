"""Re-grade the key models under the primary any-outcome label (user decision 2026-10-13), harm label beside.

Selection sets: H2 (varied scenes), H_T2 (push past tall objects), H (three-block). BOTH readout throughout; the
three-block US also with hold-30 + its validation floor (its confirmed readout).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

RESULT = ROOT / "results/jenga/any_outcome_regrade.json"


def events_from_objects(end, start):
    """end (64, N, 7), start (N, 7) -> (64, N, 3) bool: tilt change >= 45 deg, off table, moved >= 30 mm."""
    import phase2_scenes as ps
    tilt = np.abs(ps.tilt_deg(end[..., 3:]) - ps.tilt_deg(start[None, :, 3:])) >= 45
    off = (end[..., 0] < 0.3) | (end[..., 0] > 0.7) | (np.abs(end[..., 1]) > 0.4) | (end[..., 2] < ps.TABLE_TOP - 0.01)
    mov = np.linalg.norm(end[..., :3] - start[None, :, :3], axis=-1) >= 0.03
    return np.stack([tilt, off, mov], -1)


def events_three_block(traces):
    """traces (64, 38, 61): block positions [0:9], first two rotation columns [9:27]; reference = first step."""
    def pose(s):
        pos = s[..., :9].reshape(*s.shape[:-1], 3, 3)
        cols = s[..., 9:27].reshape(*s.shape[:-1], 3, 3, 2)
        z = np.cross(cols[..., 0], cols[..., 1])                # body z axis in world frame
        tilt = np.degrees(np.arccos(np.clip(z[..., 2], -1, 1)))
        return pos, tilt
    p0, t0 = pose(traces[:, 0]); p1, t1 = pose(traces[:, -1])
    tilt = np.abs(t1 - t0) >= 45
    mov = np.linalg.norm(p1 - p0, axis=-1) >= 0.03
    off = p1[..., 2] < 0.40                                  # below the table top
    return np.stack([tilt, off, mov], -1)


def labels(ev):
    cnt = ev.sum(0)
    fork = bool(((cnt >= 2) & (cnt <= 62)).any())
    safe = bool(((cnt == 0) | (cnt == 64)).all())
    return fork, safe


def main():
    import torch
    import phase1_models as p1
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import phase2_models as pm
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um  # noqa: F401
    import visual_pipeline_panel3 as p3  # noqa: F401
    from fresh_fragility_panel import topple_count
    from phase2_amend4 import alarm, score
    out = {}

    def grade(name, rows):
        harm = np.array([r["harm"] for r in rows]); af = np.array([r["any_fork"] for r in rows]); asafe = np.array([r["any_safe"] for r in rows])
        keep = af | asafe
        res = {"states": len(rows), "harm_forks": int(harm.sum()), "any_forks": int(af.sum()), "any_safe": int(asafe.sum()),
               "excluded_1_or_63": int((~keep).sum())}
        for m in rows[0]["alarms"]:
            a = np.array([r["alarms"][m] for r in rows])
            res[m] = {"harm_label": score(a, harm), "any_outcome_label": score(a[keep], af[keep])}
        out[name] = res
        RESULT.write_text(json.dumps(out, indent=1) + "\n")
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv["score"] for kk, vv in v.items()}) for k, v in res.items()}), flush=True)

    def hybrid(path_model, det_path):
        s = torch.load(path_model, map_location="cuda")
        h = hy.build_head(s["action_dim"]).cuda(); h.load_state_dict(s["state_dict"]); h.eval()
        d = wm.WorldModel.load(det_path); d.net.eval()
        return h, d, s["residual_scale"]

    # ---------------- H2 (varied scenes)
    P2 = pm.OUT
    te = pm.GaussTE(torch.load(P2 / "Gauss_TE_phase2.pt"))
    dets = {a: wm.WorldModel.load(P2 / f"model_{a}.pt") for a in ("US2_xlong", "US2_x3data")}
    hybs = {"HYB": hybrid(P2 / "model_HYB.pt", P2 / "model_US2_xlong.pt"), "HYB_x3": hybrid(P2 / "model_HYB_x3.pt", P2 / "model_US2_x3data.pt")}
    rows = []
    for n, path in enumerate(sorted((P2 / "h2").glob("*.npz"))):
        with np.load(path) as f:
            ev = events_from_objects(f["traces"][:, -1], f["start_state"]); k = int(f["harm"].sum())
            z0 = torch.as_tensor(f["z0"], device="cuda").float(); acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
            real = f["z"].astype(np.float64).reshape(64, 2, -1)
        af, asafe = labels(ev)
        al = {"real_futures": bool(alarm(real[:, 0], real[:, 1], "BOTH"))}
        g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n)
        st = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
        for a, m in dets.items():
            al[a] = bool(alarm(*sm.roll(m, st, acts), "BOTH"))
        for a, (h, d, rs) in hybs.items():
            g2 = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
            fut = hy.sample(h, d, z0[None].expand(64, -1, -1, -1).contiguous(), d._act(torch.as_tensor(acts, device="cuda")), rs, g2)
            al[a] = bool(alarm(fut[:, 0].reshape(64, -1).double().cpu().numpy(), fut[:, 1].reshape(64, -1).double().cpu().numpy(), "BOTH"))
        rows.append({"harm": 2 <= k <= 62, "any_fork": af, "any_safe": asafe, "alarms": al})
        torch.cuda.empty_cache()
    grade("H2_varied", rows)

    # ---------------- H_T2 (push past tall objects)
    T2 = ROOT / "results/program_b/t2"
    te2 = pm.GaussTE(torch.load(T2 / "Gauss_TE_phase2.pt")); m2 = wm.WorldModel.load(T2 / "model_US2_long.pt")
    rows = []
    for n, path in enumerate(sorted((T2 / "h2").glob("*.npz"))):
        with np.load(path) as f:
            ev = events_from_objects(f["traces"][:, -1], f["start_state"]); k = int(f["harm"].sum())
            z0 = torch.as_tensor(f["z0"], device="cuda").float(); acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
            real = f["z"].astype(np.float64).reshape(64, 2, -1)
        af, asafe = labels(ev)
        g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n)
        al = {"real_futures": bool(alarm(real[:, 0], real[:, 1], "BOTH")),
              "US_T2": bool(alarm(*sm.roll(m2, te2(z0[None].expand(64, -1, -1, -1).contiguous(), g), acts), "BOTH"))}
        rows.append({"harm": 2 <= k <= 62, "any_fork": af, "any_safe": asafe, "alarms": al})
        torch.cuda.empty_cache()
    grade("H_T2_push", rows)

    # ---------------- H (three-block)
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    us, tau = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]
    gop = sm.G.load(sm.DATA / "G.pt")
    rows = []
    with torch.no_grad():
        for path in sorted((p1.OUT / "h").glob("ep*.npz")):
            with np.load(path) as f:
                ev = events_three_block(f["traces"]); k = topple_count(f["traces"], f["start_tilt"])
                d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
                z0 = torch.as_tensor(f["z0"], device="cuda").float()
                e10, e30 = sm.roll(us, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()), p1d.augment(f["actions"], None, False))
                real = f["z"].astype(np.float64).reshape(64, 2, -1)
            af, asafe = labels(ev)
            al = {"real_futures": bool(alarm(real[:, 0], real[:, 1], "BOTH")), "US_BOTH": bool(alarm(e10, e30, "BOTH")),
                  "US_H30_floor": bool(alarm(e10, e30, "H30_floor", tau))}
            rows.append({"harm": 2 <= k <= 62, "any_fork": af, "any_safe": asafe, "alarms": al})
            torch.cuda.empty_cache()
    grade("H_three_block", rows)


if __name__ == "__main__":
    main()
