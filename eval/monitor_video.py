"""~2 minutes of the three-block task (pick the middle block), replayed episode after episode on fresh block layouts,
with the deployed monitor running at every chunk start (every 8 actions = 0.8 s). Monitoring only, no filter.

Monitor = deployed three-block monitor: close-camera frame -> DINOv2/PCA latent -> G start sampler (64 perception
draws) -> US dynamics with 64 execution-noise snippets -> hold-30 split test with the validation floor (H30_floor).
Ground truth (for display only) = 64 MuJoCo futures from the true state with the corrected (valid) perception error
and the same execution-noise snippets: neighbour topples and any-outcome events.

Stages:  sim (CPU pool)  ->  monitor (GPU, one process)  ->  compose (video)
Episodes: RESET_BASE 14000 + episode id (layouts never used before). Candidates are simulated, then episodes are
chosen to fill ~2 min: those with ground-truth forks first (to show fragile moments), the rest in random order.
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

OUT = ROOT / "results/program_b/monitor_video"
VIDEO = ROOT / "results/program_b/monitor_video_3block.mp4"
RESET_BASE, N_CANDIDATES, TARGET_ACTIONS = 14000, 16, 1200
SEEDS = {"pick": 20261401, "truth": 20261402, "monitor": 20261403}


def sim_job(e):
    import traceback
    try:
        return e, _sim(e), None
    except Exception:
        return e, None, traceback.format_exc()


def _sim(e):
    import mujoco
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    from regrade_any_outcome import events_three_block, labels
    out = OUT / f"ep{e}.npz"
    if out.exists():
        return True
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(e).actions, np.float32); replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="mv_"))))
    r_close = mujoco.Renderer(sim.model, height=384, width=384); cam = close_camera(mujoco)
    r_wide = mujoco.Renderer(sim.model, height=360, width=480)
    starts = [s for s in chunk_starts(len(actions)) if s + 8 <= len(actions)]
    close, wide, chunk_info = [], [], {}
    try:
        sim.reset(int(e) + RESET_BASE)
        for i, a in enumerate(actions):
            if i in starts:
                snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
                tilt = sim.block_diagnostics()["tilt"][1:].copy()
                rest = pr.blocks_at_rest(sim)
                r_close.update_scene(sim.data, camera=cam); frame = r_close.render().copy()
                chunk = actions[i:i + 8]; key = (int(e) + RESET_BASE, i)
                traces = [um.rollout(sim, snap, chunk, snippets[j], [errors.sample(SEEDS["truth"], *key, j)]) for j in range(64)]
                ev = events_three_block(np.stack(traces)); af, asafe = labels(ev)
                chunk_info[i] = {"frame": frame, "chunk": chunk, "topples": topple_count(traces, tilt), "any_fork": af,
                                 "any_safe": asafe, "at_rest": rest}
                sim.restore(snap); sim.data.qacc_warmstart[:] = warm
            r_close.update_scene(sim.data, camera=cam); close.append(r_close.render()[::2, ::2].copy())
            r_wide.update_scene(sim.data, camera="cam_fixed"); wide.append(r_wide.render().copy())
            sim.execute(a)
        r_close.update_scene(sim.data, camera=cam); close.append(r_close.render()[::2, ::2].copy())
        r_wide.update_scene(sim.data, camera="cam_fixed"); wide.append(r_wide.render().copy())
        diag = sim.block_diagnostics()
    finally:
        r_close.close(); r_wide.close(); sim.close()
    ss = sorted(chunk_info)
    np.savez_compressed(out.with_suffix(".tmp.npz"), close=np.stack(close), wide=np.stack(wide), starts=np.array(ss),
                        start_frames=np.stack([chunk_info[s]["frame"] for s in ss]),
                        chunks=np.stack([chunk_info[s]["chunk"] for s in ss]),
                        topples=np.array([chunk_info[s]["topples"] for s in ss]),
                        any_fork=np.array([chunk_info[s]["any_fork"] for s in ss]),
                        any_safe=np.array([chunk_info[s]["any_safe"] for s in ss]),
                        at_rest=np.array([chunk_info[s]["at_rest"] for s in ss]),
                        final_tilt=np.asarray(diag["tilt"]))
    out.with_suffix(".tmp.npz").replace(out)
    return True


def stage_sim(workers=8):
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    OUT.mkdir(parents=True, exist_ok=True)
    replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
    cand = sorted(np.random.default_rng(SEEDS["pick"]).choice(ids, N_CANDIDATES, replace=False).tolist(), key=int)
    (OUT / "candidates.json").write_text(json.dumps(cand) + "\n")
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (e, ok, err) in enumerate(pool.imap_unordered(sim_job, cand), 1):
            print(f"sim {n}/{len(cand)} ep{e}", "FAILED\n" + err if err else "", flush=True)


def truth2_job(e):
    """Ground truth with the gripper-relative target events (label revision 2026-10-13), same futures as stage sim."""
    import traceback
    try:
        import uncertainty_monitor_dev as um
        import visual_pipeline_panel3 as p3
        from jenga_runtime import DEFAULT_LMDB, JengaReplay
        from jenga_short_held_tails import DirectJengaSim, extract_sim
        from regrade_any_outcome import events_three_block_v2, labels
        target = OUT / f"ep{e}.truth2.json"
        if target.exists():
            return e, None
        errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
        replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(e).actions, np.float32); replay.close()
        with np.load(OUT / f"ep{e}.npz") as f:
            starts = [int(s) for s in f["starts"]]
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="mv2_"))))
        out = {}
        try:
            sim.reset(int(e) + RESET_BASE)
            for i, a in enumerate(actions):
                if i in starts:
                    snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy(); key = (int(e) + RESET_BASE, i)
                    tr = np.stack([um.rollout(sim, snap, actions[i:i + 8], snippets[j], [errors.sample(SEEDS["truth"], *key, j)]) for j in range(64)])
                    af, asafe = labels(events_three_block_v2(tr)); out[i] = {"fork": bool(af), "safe": bool(asafe)}
                    sim.restore(snap); sim.data.qacc_warmstart[:] = warm
                if i > max(starts):
                    break
                sim.execute(a)
        finally:
            sim.close()
        target.write_text(json.dumps(out) + "\n")
        return e, None
    except Exception:
        return e, traceback.format_exc()


def stage_truth2(workers=9):
    eps = json.loads((ROOT / "results/jenga/monitor_video_summary.json").read_text())["episodes"]
    with mp.get_context("spawn").Pool(workers) as pool:
        for e, err in pool.imap_unordered(truth2_job, eps):
            print("truth2", e, err or "", flush=True)


def stage_monitor():
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from outcome_modes import two_mode_test
    from phase2_amend4 import alarm
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    model, tau = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]
    g, latent, errors = sm.G.load(sm.DATA / "G.pt"), Latent(), p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    for path in sorted(OUT.glob("ep*.npz")):
        if path.name.endswith(".mon.npz") or path.name.endswith(".tmp.npz"):
            continue
        target = path.with_suffix(".mon.npz")
        if target.exists():
            continue
        e = int(path.stem[2:])
        with np.load(path) as f:
            frames, chunks, starts = f["start_frames"], f["chunks"], f["starts"]
        alarms, proj, groups, minority, sep, ms = [], [], [], [], [], []
        with torch.no_grad():
            for frame, chunk, s in zip(frames, chunks, starts):
                import time
                t0 = time.time()
                z0 = torch.as_tensor(latent(frame[None])[0], device="cuda").float()
                d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                           (errors.sample(SEEDS["monitor"], e + RESET_BASE, int(s), j) for j in range(64))]), dtype=torch.float32)
                e10, e30 = sm.roll(model, g(z0[None].expand(64, -1, -1, -1), d), wm.panel_actions(chunk, snippets))
                a = bool(alarm(e10, e30, "H30_floor", tau))
                torch.cuda.synchronize(); ms.append((time.time() - t0) * 1000)
                t = two_mode_test(e30, 0.0)
                c = e30 - e30.mean(0); u, sv, vt = np.linalg.svd(c, full_matrices=False)
                alarms.append(a); proj.append(c @ vt[:2].T / np.sqrt(e30.shape[1] // 3)); groups.append(np.asarray(t.labels))
                minority.append(int(t.minority))
                sep.append(float(np.linalg.norm(e30[t.labels == 0].mean(0) - e30[t.labels == 1].mean(0)) / np.sqrt(e30.shape[1] // 3))
                           if 0 < t.labels.sum() < 64 else 0.0)
                torch.cuda.empty_cache()
        np.savez(target, alarm=np.array(alarms), proj=np.stack(proj), groups=np.stack(groups), minority=np.array(minority),
                 separation=np.array(sep), latency_ms=np.array(ms), tau=tau)
        print("monitored", path.stem, f"{np.median(ms):.0f} ms/decision", flush=True)


def choose():
    eps = []
    for path in sorted(OUT.glob("ep*.npz")):
        if path.name.count(".") > 1:
            continue
        with np.load(path) as f:
            eps.append((int(path.stem[2:]), len(f["close"]) - 1, int(f["any_fork"].sum())))
    rng = np.random.default_rng(SEEDS["pick"] + 1)
    with_forks = [x for x in eps if x[2] > 0]; rest = [x for x in eps if x[2] == 0]
    order = sorted(with_forks, key=lambda x: -x[2]) + [rest[i] for i in rng.permutation(len(rest))]
    chosen, total = [], 0
    for x in order:
        if total >= TARGET_ACTIONS:
            break
        chosen.append(x); total += x[1]
    return [x[0] for x in sorted(chosen, key=lambda x: x[0])]


def stage_compose():
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1280, 720
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16); big = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 13); mid = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
    except OSError:
        font = big = small = mid = ImageFont.load_default()
    chosen = choose()
    writer = imageio_ffmpeg.write_frames(str(VIDEO), (W, H), fps=10, codec="libx264", quality=8, macro_block_size=1)
    writer.send(None)
    summary = {"episodes": chosen, "decisions": 0, "truth_fork": 0, "truth_safe": 0, "excluded": 0,
               "caught": 0, "missed": 0, "false_alarm": 0, "quiet": 0, "latency_ms": []}
    clock = 0.0
    for e in chosen:
        f = dict(np.load(OUT / f"ep{e}.npz")); m = dict(np.load(OUT / f"ep{e}.mon.npz"))
        t2 = OUT / f"ep{e}.truth2.json"
        if t2.exists():                                   # gripper-relative target events (label revision)
            lab = json.loads(t2.read_text())
            f["any_fork"] = np.array([lab[str(int(s))]["fork"] for s in f["starts"]])
            f["any_safe"] = np.array([lab[str(int(s))]["safe"] for s in f["starts"]])
        starts = list(f["starts"]); n = len(f["close"])
        timeline = []                                                   # (start, alarm, truth) for the strip
        for k, s in enumerate(starts):
            truth = "fork" if f["any_fork"][k] else ("safe" if f["any_safe"][k] else "unclear")
            timeline.append((s, bool(m["alarm"][k]), truth, int(f["topples"][k]), bool(f["at_rest"][k])))
            if f["at_rest"][k] and truth != "unclear":
                summary["decisions"] += 1; summary["truth_" + truth] += 1
                key = {(True, "fork"): "caught", (False, "fork"): "missed", (True, "safe"): "false_alarm", (False, "safe"): "quiet"}
                summary[key[(bool(m["alarm"][k]), truth)]] += 1
            else:
                summary["excluded"] += 1
        summary["latency_ms"] += m["latency_ms"].tolist()
        for t in range(n):
            cur = max([k for k, s in enumerate(starts) if s <= t], default=None)
            img = Image.new("RGB", (W, H), (246, 247, 249)); d = ImageDraw.Draw(img)
            img.paste(Image.fromarray(f["wide"][t]).resize((560, 420)), (16, 56))
            img.paste(Image.fromarray(f["close"][t]).resize((300, 300)), (592, 56))
            d.text((16, 14), "Three-block task: pick the middle block   |   monitor only (no filter)", fill=(20, 28, 40), font=big)
            d.text((16, 480), "scene camera", fill=(90, 100, 110), font=small)
            d.text((592, 360), "monitor input: close camera only", fill=(90, 100, 110), font=small)
            d.text((16, 500), f"episode {e} (fresh layout)    t = {clock + t * 0.1:5.1f} s    action {t}/{n - 1}", fill=(20, 28, 40), font=font)
            # current decision panel
            x0 = 908
            if cur is not None:
                s = starts[cur]; a = bool(m["alarm"][cur]); truth = timeline[cur][2]
                inside = t < s + 8
                col = (200, 60, 40) if a else (30, 140, 80)
                d.rectangle([x0, 56, W - 16, 120], fill=col)
                d.text((x0 + 12, 64), "FRAGILE: outcomes split" if a else "OK: one outcome", fill=(255, 255, 255), font=mid)
                d.text((x0 + 12, 96), f"decision for actions {s}-{s + 7}" + ("  (executing)" if inside else "  (done)"), fill=(255, 255, 255), font=small)
                # scatter of 64 predicted hold-30 futures
                P = m["proj"][cur]; gr = m["groups"][cur]
                sx0, sy0, sw, sh = x0, 132, W - 16 - x0, 230
                d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], fill=(255, 255, 255), outline=(210, 214, 220))
                span = max(float(np.abs(P).max()), 1e-6) * 1.15
                minor = 1 if gr.sum() <= 32 else 0
                for p_, g_ in zip(P, gr):
                    px = sx0 + sw / 2 + p_[0] / span * sw / 2; py = sy0 + sh / 2 - p_[1] / span * sh / 2
                    c = (200, 90, 30) if (g_ == minor and a) else (40, 90, 160)
                    d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=c)
                d.text((sx0 + 8, sy0 + 6), "64 imagined futures (3 s ahead)", fill=(60, 70, 80), font=small)
                d.text((sx0 + 8, sy0 + sh - 20), f"split {int(m['minority'][cur])}/64, separation {m['separation'][cur]:.2f} (floor {float(m['tau']):.2f})", fill=(60, 70, 80), font=small)
                tt = {"fork": f"TRUTH: fork\n{timeline[cur][3]}/64 real futures topple a neighbour" if timeline[cur][3] else "TRUTH: fork\nthe target drops or tips in some futures",
                      "safe": "TRUTH: safe\nall 64 real futures agree", "unclear": "TRUTH: unclear\n1 of 64 real futures differs"}[truth]
                if not timeline[cur][4]:
                    tt = "TRUTH: n/a\nblocks still moving at this start"
                d.multiline_text((x0, 368), tt, fill=(20, 28, 40), font=font, spacing=4)
                verdict = {(True, "fork"): ("caught", (200, 60, 40)), (False, "fork"): ("missed", (160, 110, 0)),
                           (True, "safe"): ("false alarm", (160, 110, 0)), (False, "safe"): ("correct: quiet", (30, 140, 80))}.get((a, truth))
                if verdict and timeline[cur][4]:
                    d.text((x0, 414), "monitor vs truth: " + verdict[0], fill=verdict[1], font=font)
                d.text((x0, 440), f"decision time: {m['latency_ms'][cur]:.0f} ms (RTX 3090)", fill=(90, 100, 110), font=small)
            # timeline strip
            ty = 560
            d.text((16, ty - 26), "decisions this episode (top: monitor, bottom: ground truth)   red = fragile / fork, green = OK / safe, grey = n/a",
                   fill=(60, 70, 80), font=small)
            bw = (W - 32) / max(n, 1)
            for (s, a, truth, tp, rest) in timeline:
                xa, xb = 16 + s * bw, 16 + (s + 8) * bw - 2
                d.rectangle([xa, ty, xb, ty + 24], fill=(200, 60, 40) if a else (30, 140, 80))
                tc = (150, 155, 160) if (not rest or truth == "unclear") else ((200, 60, 40) if truth == "fork" else (30, 140, 80))
                d.rectangle([xa, ty + 30, xb, ty + 54], fill=tc)
            cx = 16 + t * bw
            d.line([cx, ty - 4, cx, ty + 58], fill=(20, 28, 40), width=3)
            d.text((16, ty + 64), "Fragile = the monitor's 64 imagined futures split into two separated groups (outcome depends on unobservable "
                   "perception/execution error).", fill=(90, 100, 110), font=small)
            d.text((16, ty + 82), "Ground truth = 64 MuJoCo futures from the true state with the same kinds of error; shown for evaluation only, "
                   "the monitor never sees it.", fill=(90, 100, 110), font=small)
            writer.send(np.asarray(img))
        clock += (n - 1) * 0.1
    writer.close()
    lat = summary.pop("latency_ms")
    summary["latency_ms_median"] = float(np.median(lat)); summary["seconds"] = round(clock, 1)
    (ROOT / "results/jenga/monitor_video_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    {"sim": stage_sim, "monitor": stage_monitor, "truth2": stage_truth2, "compose": stage_compose}[sys.argv[1]]()
