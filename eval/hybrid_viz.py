"""Visual analysis of the best varied-scene model (HYB_x3) on H2: outcome categories, example plots and renders.

Stage 1 (GPU, this machine): futures for all H2 states -> categories; per chosen example a PCA scatter of real vs
hybrid vs deterministic-only futures. Stage 2 (CPU/EGL): re-simulate and render real outcomes for the examples.
Outputs go to the page folder given on the command line.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_models as pm  # noqa: E402


def stage_futures(outdir):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phase2_hybrid as hy
    import step3_world_model as wm
    from outcome_modes import two_mode_test
    from phase2_amend4 import alarm
    s = torch.load(pm.OUT / "model_HYB_x3.pt", map_location="cuda")
    head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
    det = wm.WorldModel.load(pm.OUT / "model_US2_x3data.pt"); det.net.eval()
    rows, store = [], {}
    for n, path in enumerate(sorted((pm.OUT / "h2").glob("*.npz"))):
        with np.load(path) as f:
            k = int(f["harm"].sum()); harm = f["harm"].astype(bool)
            z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(64, -1, -1, -1).contiguous()
            act = det._act(torch.as_tensor(np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, -1), device="cuda"))
            real = f["z"].reshape(64, 2, -1).astype(np.float64)
        g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
        fut = hy.sample(head, det, z0, act, s["residual_scale"], g)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zd = hy.det_endpoints(det, z0, act).float()
        e10 = fut[:, 0].reshape(64, -1).double().cpu().numpy(); e30 = fut[:, 1].reshape(64, -1).double().cpu().numpy()
        d30 = zd[:, 1].reshape(64, -1).double().cpu().numpy()
        a = bool(alarm(e10, e30, "BOTH")); fork = 2 <= k <= 62
        cat = ("caught" if a else "missed") if fork else ("false_alarm" if a else "quiet")
        t30 = two_mode_test(e30, 0.0)
        rows.append({"state": path.stem, "family": path.stem.split("_")[0], "harm_count": k, "fork": fork, "alarm": a,
                     "category": cat, "model_minority": int(t30.minority), "model_ashman_d": float(t30.ashman_d) if np.isfinite(t30.ashman_d) else None,
                     "spread_ratio": float(e30.std(0).sum() / max(real[:, 1].std(0).sum(), 1e-9))})
        store[path.stem] = (real[:, 1], e30, d30, harm)
        torch.cuda.empty_cache()
    # choose examples: per category, the clearest cases (forks: most balanced harm counts first; safe: random-but-fixed)
    rng = np.random.default_rng(7)
    chosen = []
    for cat, n_pick in (("caught", 3), ("missed", 3), ("false_alarm", 2), ("quiet", 1)):
        rs = [r for r in rows if r["category"] == cat]
        if cat in ("caught", "missed"):
            rs = sorted(rs, key=lambda r: abs(r["harm_count"] - 32))
            pick, fams = [], set()
            for r in rs:                                  # diversify families
                if r["family"] not in fams or len(pick) >= 2:
                    pick.append(r); fams.add(r["family"])
                if len(pick) == n_pick:
                    break
        else:
            pick = [rs[i] for i in rng.permutation(len(rs))[:n_pick]]
        chosen += pick
    plt.rcParams.update({"font.size": 10})
    for r in chosen:
        real30, e30, d30, harm = store[r["state"]]
        mu = real30.mean(0); u, sv, vt = np.linalg.svd(real30 - mu, full_matrices=False)
        P = vt[:2].T
        pr, pe, pd = (real30 - mu) @ P, (e30 - mu) @ P, (d30 - mu) @ P
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True, sharey=True)
        ax[0].scatter(pr[~harm, 0], pr[~harm, 1], s=14, c="#127c87", label="no harm"); ax[0].scatter(pr[harm, 0], pr[harm, 1], s=14, c="#c4661a", label="harm")
        ax[0].set_title("real futures (64)"); ax[0].legend(frameon=False, fontsize=8, loc="best")
        ax[1].scatter(pr[:, 0], pr[:, 1], s=10, c="#cfd6dc"); ax[1].scatter(pd[:, 0], pd[:, 1], s=14, c="#5d6b79")
        ax[1].set_title("deterministic stream only")
        ax[2].scatter(pr[:, 0], pr[:, 1], s=10, c="#cfd6dc"); ax[2].scatter(pe[:, 0], pe[:, 1], s=14, c="#1b2430")
        ax[2].set_title("hybrid (stream + flow head)")
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False); a_.set_xlabel("PC1 of real futures")
        ax[0].set_ylabel("PC2")
        fig.tight_layout(); fig.savefig(outdir / f"scatter_{r['state']}.png", dpi=110); plt.close(fig)
    summary = {"counts": {c: sum(r["category"] == c for r in rows) for c in ("caught", "missed", "false_alarm", "quiet")},
               "missed_reasons": {"model_minority_lt2": sum(1 for r in rows if r["category"] == "missed" and r["model_minority"] < 2),
                                  "model_groups_not_separated": sum(1 for r in rows if r["category"] == "missed" and (r["model_ashman_d"] or 0) <= 2)},
               "spread_ratio_median": {c: float(np.median([r["spread_ratio"] for r in rows if r["category"] == c])) for c in ("caught", "missed", "false_alarm", "quiet")},
               "harm_count_median": {c: float(np.median([r["harm_count"] for r in rows if r["category"] == c])) for c in ("caught", "missed", "false_alarm", "quiet")},
               "chosen": chosen, "rows": rows}
    (outdir / "hybrid_viz.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "chosen")}, indent=1), flush=True)


def stage_render(outdir):
    """Start frame + end frame of one harm and one no-harm real future (or two futures for safe states)."""
    import mujoco
    from PIL import Image
    import phase2_data as pdata
    import phase2_scenes as ps
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    info = json.loads((outdir / "hybrid_viz.json").read_text())
    for r in info["chosen"]:
        fam, seed, s = r["state"].split("_"); seed, index = int(seed), int(s[1:])
        with np.load(pm.OUT / "h2" / f"{r['state']}.npz") as f:
            harm = f["harm"].astype(bool); Image.fromarray(f["close_frame"]).save(outdir / f"start_{r['state']}.png")
        want = {}
        if harm.any():
            want[int(np.flatnonzero(harm)[0])] = "harm"
        if (~harm).any():
            want[int(np.flatnonzero(~harm)[0])] = "noharm"
        scene, sim, acts, starts = pdata.prepare(fam, seed)
        snippets = np.load(um.BENCH)["snippets"]; pos_cov, rot_cov = ps.pooled_error_model()
        rend = mujoco.Renderer(sim.model, height=384, width=384); cam = close_camera(mujoco)
        for i, a in enumerate(acts):
            if i == index:
                break
            sim.execute(a)
        snap = sim.snapshot(); chunk = acts[index:index + 8]
        for j in range(max(want) + 1):
            err = ps.draw_error(len(scene["objects"]), pos_cov, rot_cov, pm.SEEDS["h2_futures"], pdata.FAMILY_CODE[fam], seed, index, j)
            sim.restore(snap); sim.perturb(*err)
            a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= snippets[j]
            for c in list(a) + [chunk[-1]] * 30:
                sim.execute(c)
            if j in want:
                rend.update_scene(sim.data, camera=cam)
                Image.fromarray(rend.render().copy()).save(outdir / f"end_{want[j]}_{r['state']}.png")
        rend.close()
        print("rendered", r["state"], flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    {"futures": stage_futures, "render": stage_render}[sys.argv[1]](out)
