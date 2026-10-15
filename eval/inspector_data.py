"""Data for the dashboard's Inspector tab: paired closed-loop episodes (policy alone vs policy + filter v2) replayed from
their saved decision states, with per-decision ground truth and the monitor's imagined futures.

For each selected episode and arm:
  video   fixed camera | monitor's close camera, every control step, banner with the decision (replayed from the saved
          state at each decision and the chunk actually executed, with the run's execution-noise sequence and physics)
  rows    per decision: time, flag, filter action, offset, real outcome of the original and the executed chunk
          (64 futures each: which ones topple), the monitor's 64 imagined futures for both (2-D projection + groups)
Output: results/program_b/inspector/{episodes.json, video/<arm>_ep<i>.mp4}

usage: HARD_MASS=0.3 python3 eval/inspector_data.py [n_episodes]
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / os.environ.get("INSP_OUT", "results/program_b/inspector")
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"alone": os.environ.get("INSP_ALONE", "paired_monitor"), "filter": os.environ.get("INSP_FILTER", "paired_cem_v2")}
EXEC_SCALE = float(os.environ.get("EXEC_NOISE_SCALE", 1.0))      # the run's execution-noise level (final test: 2)
SEED = 20261550


def pick(n):
    """Episodes where the arms differ (filter prevented or caused a topple), then both toppled, then neither."""
    load = lambda tag: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{tag}.jsonl").read_text().splitlines()}
    a, f = load(ARMS["alone"]), load(ARMS["filter"])
    eps = sorted(set(a) & set(f))
    cat = {e: ("prevented" if a[e]["neighbour_toppled"] and not f[e]["neighbour_toppled"] else
               "caused" if f[e]["neighbour_toppled"] and not a[e]["neighbour_toppled"] else
               "both toppled" if a[e]["neighbour_toppled"] else "neither") for e in eps}
    order = [e for c in ("prevented", "caused", "both toppled", "neither") for e in eps if cat[e] == c]
    quota = {"prevented": n * 2 // 5, "caused": n * 2 // 5, "both toppled": n // 10, "neither": n - 2 * (n * 2 // 5) - n // 10}
    chosen, used = [], {k: 0 for k in quota}
    for e in order:
        if used[cat[e]] < quota[cat[e]]:
            chosen.append(e); used[cat[e]] += 1
    return chosen, cat, a, f


def _sim():
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="insp_"))))
    ms = float(os.environ.get("HARD_MASS", 0.3))
    for name in (("block_left", "block_right") if ms else ()):
        b = sim.model.body(name).id
        sim.model.body_mass[b] *= ms; sim.model.body_inertia[b] *= ms
    return sim


def truth_job(args):
    """Real outcome of one chunk from one saved state: which of 64 futures topple a neighbour."""
    key, state, ctrl, target, gripper, tilt, chunk = args
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import TOPPLE_DEG, neighbour_tilts
    global _S
    if "_S" not in globals():
        _S = _sim()
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    flags = []
    for j in range(int(os.environ.get("TRUTH_N", 64))):              # 64 by default; fewer for quick screening
        tr = um.rollout(_S, snap, chunk, snippets[j] * EXEC_SCALE, [errors.sample(SEED, *key, j)])
        flags.append(bool(np.any((neighbour_tilts(tr[-1]) >= TOPPLE_DEG) & (np.asarray(tilt) < TOPPLE_DEG))))
    return key, flags


def video_job(args):
    """Replay one arm's episode from its saved decisions with the run's noise sequence; render both cameras."""
    arm, ep, path = args
    import mujoco
    import imageio_ffmpeg
    import uncertainty_monitor_dev as um
    from PIL import Image, ImageDraw
    from jenga_render_frames import close_camera
    with np.load(path) as f:
        S = {k: f[k] for k in f.files}
    snippets = np.load(um.BENCH)["snippets"]
    nrng = np.random.default_rng([20261502, ep])
    first = int(S["chunk_index"][0])
    noises = [snippets[nrng.integers(len(snippets))] * EXEC_SCALE for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = _sim()
    rf = mujoco.Renderer(sim.model, height=240, width=320); rc = mujoco.Renderer(sim.model, height=240, width=240)
    cam = close_camera(mujoco)
    out = OUT / "video" / f"{arm}_ep{ep:03d}.mp4"
    w = imageio_ffmpeg.write_frames(str(out), (560, 260), fps=10, codec="libx264", quality=7, macro_block_size=1)
    w.send(None)
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d])
        if S["abstained"][d]:
            txt, col = "paused (nothing safe nearby)", (107, 90, 166)
        elif S["intervened"][d]:
            txt, col = "filter changed the action", (215, 130, 20)
        elif S["alarm"][d]:
            txt, col = "monitor flag", (194, 73, 29)
        else:
            txt, col = "ok", (47, 125, 79)
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]
            sim.execute(a)
            rf.update_scene(sim.data, camera="cam_fixed"); rc.update_scene(sim.data, camera=cam)
            im = Image.new("RGB", (560, 260), (20, 26, 24))
            im.paste(Image.fromarray(rf.render()), (0, 20)); im.paste(Image.fromarray(rc.render()), (320, 20))
            dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 560, 19], fill=col)
            dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   decision {d}   {txt}", fill=(255, 255, 255))
            w.send(np.asarray(im))
    w.close(); rf.close(); rc.close(); sim.close()
    return arm, ep, int(first)


def imagined(mon, sim, S, d, chunks):
    """Monitor's imagined hold-30 futures for the given chunks at saved decision d: 2-D projection (shared basis) + groups."""
    import cem_filter as cf
    sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
    st = mon.starts(mon.frame(sim))
    _, e30 = mon.endpoints(st, np.stack(chunks))
    ok, lab, sep = cf.split_test_gpu(e30)
    X = e30.double().cpu().numpy()
    mu = X[0].mean(0); u, s_, vt = np.linalg.svd(X[0] - mu, full_matrices=False)
    P = [((x - mu) @ vt[:2].T / np.sqrt(x.shape[1] // 3)).round(3).tolist() for x in X]
    return P, lab.cpu().numpy().tolist(), [bool(v) for v in ok.cpu().numpy()], [float(v) for v in sep.cpu().numpy()]


def main(n=20):
    import cem_filter as cf
    (OUT / "video").mkdir(parents=True, exist_ok=True)
    chosen, cat, a_res, f_res = pick(n)
    data = {"episodes": [], "categories": cat}
    S_all = {}
    jobs_v, jobs_t = [], []
    for arm, tag in ARMS.items():
        for ep in chosen:
            p = RUNS / tag / f"ep{ep:03d}.npz"
            if not p.exists():
                continue
            S = dict(np.load(p)); S_all[(arm, ep)] = S
            jobs_v.append((arm, ep, str(p)))
            for d in range(len(S["chunk_index"])):
                base = (S["state"][d], S["ctrl"][d], S["target"][d], S["gripper"][d], S["start_tilt"][d])
                jobs_t.append(((ep, 0 if arm == "alone" else 1, d * 2), *base, S["nominal"][d]))
                if S["intervened"][d] or S["abstained"][d]:
                    jobs_t.append(((ep, 0 if arm == "alone" else 1, d * 2 + 1), *base, S["executed"][d]))
    workers = int(os.environ.get("INSP_WORKERS", 24))
    truth = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n_, (key, flags) in enumerate(pool.imap_unordered(truth_job, jobs_t), 1):
            truth[key] = flags
            if n_ % 200 == 0:
                print(f"truth {n_}/{len(jobs_t)}", flush=True)
        for arm, ep, _ in pool.imap_unordered(video_job, jobs_v):
            print("video", arm, ep, flush=True)
    mon, sim = cf.Monitor(seed=20261551), _sim()
    for ep in chosen:
        entry = {"episode": ep, "category": cat[ep], "arms": {}}
        for arm in ARMS:
            if (arm, ep) not in S_all:
                continue
            S = S_all[(arm, ep)]; res = (a_res if arm == "alone" else f_res)[ep]
            rows = []
            for d in range(len(S["chunk_index"])):
                changed = bool(S["intervened"][d] or S["abstained"][d])
                chunks = [S["nominal"][d]] + ([S["executed"][d]] if changed else [])
                P, lab, ok, sep = imagined(mon, sim, S, d, chunks)
                t_nom = truth[(ep, 0 if arm == "alone" else 1, d * 2)]
                t_exe = truth.get((ep, 0 if arm == "alone" else 1, d * 2 + 1), t_nom)
                rows.append({"d": d, "chunk": int(S["chunk_index"][d]), "t": round(int(S["chunk_index"][d]) * 0.8, 1),
                             "flag": bool(S["alarm"][d]) or changed, "changed": bool(S["intervened"][d]) and not bool(S["abstained"][d]),
                             "paused": bool(S["abstained"][d]),
                             "offset_mm": (np.abs(S["executed"][d][:, :3] - S["nominal"][d][:, :3]).max() * 1000).round(1).item() if changed else 0.0,
                             "real_orig": [int(x) for x in t_nom], "real_exec": [int(x) for x in t_exe],
                             "risk_orig": round(sum(t_nom) / 64, 3), "risk_exec": round(sum(t_exe) / 64, 3),
                             "img_orig": P[0], "img_exec": P[1] if changed else None, "groups_orig": lab[0], "groups_exec": lab[1] if changed else None,
                             "split_orig": ok[0], "split_exec": ok[1] if changed else None})
            started = [c["chunk"] for c in res["log"] if c.get("topple_started")]
            first_chunk = int(S["chunk_index"][0])
            entry["arms"][arm] = {"video": f"video/{arm}_ep{ep:03d}.mp4", "toppled": bool(res["neighbour_toppled"]),
                                  "success": bool(res.get("seconds_to_3cm") is not None and not res["neighbour_toppled"]),
                                  "topple_chunk": started[0] if started else None, "first_chunk": first_chunk,
                                  "video_t0": round(first_chunk * 0.8, 1), "rows": rows}
        al, fi = entry["arms"].get("alone"), entry["arms"].get("filter")
        if al and fi:
            div = next((r["chunk"] for r in fi["rows"] if r["changed"] or r["paused"]), None)
            entry["first_divergence_chunk"] = div
        data["episodes"].append(entry)
        print("episode", ep, cat[ep], flush=True)
    (OUT / "episodes.json").write_text(json.dumps(data) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
