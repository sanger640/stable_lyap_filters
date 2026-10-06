"""Compose the results video (1920x1080, 30 fps, H.264) from paper/video/scenes/*.npz and the result files.

Descriptive only. Every number on screen is read from a result file at render time.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SCENES = HERE / "scenes"
OUT = HERE / "fork_monitor_results.mp4"
W, H, FPS = 1920, 1080, 30
INK, MUTED, LINE, CARD, BG = "#1f2328", "#5b6470", "#d0d7de", "#f6f8fa", "#ffffff"
TEAL, ORANGE, VIOLET, GREEN, RED = "#1b8a7a", "#e07a1f", "#6f42c1", "#1a7f37", "#cf222e"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": LINE,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
WM_FRAMES = list(range(1, 38, 2))
CAPTIONS = {
    "1_safe": ("A safe state",
               "All 64 real futures stay standing. The probe's and the world model's imagined futures stay together too: "
               "both correctly stay quiet."),
    "2_fork_both": ("A fork both engines catch",
                    "The real futures split into 'stays' and 'topples'. The futures imagined in MuJoCo split as well, and so "
                    "do the world model's: both raise the alarm."),
    "3_fork_wm_misses": ("A fork only the MuJoCo probe catches",
                         "Only a few of the probe's imagined futures topple, but they form a separate, persistent group, so it "
                         "alarms. The world model's 64 futures stay in one bundle: it misses the fork."),
    "4_wm_false_alarm": ("A false alarm by the world model",
                         "Nothing topples in reality or in MuJoCo. The world model's futures drift into two groups anyway: "
                         "a false alarm."),
    "5_fork_both_miss": ("A fork both engines miss",
                         "Reality splits, but the probe's perception estimate places every imagined future on the 'stays' "
                         "side, and the world model's futures do not split: both miss."),
}


# ----------------------------------------------------------------------------- helpers
def new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    return fig


def ax_px(fig, x, y, w, h, **kw):
    """Axes from pixel box (x, y from top-left)."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def text(fig, x, y, s, size=20, color=INK, weight="normal", ha="left", va="top", **kw):
    return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)


def card(fig, x, y, w, h, color=CARD, edge=LINE):
    a = ax_px(fig, x, y, w, h); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0,rounding_size=0.03",
                               fc=color, ec=edge, lw=1.5, transform=a.transAxes))
    return a


def grab(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def wrap(s, n):
    import textwrap
    return "\n".join(textwrap.wrap(s, n))


def load_scene(name):
    s = dict(np.load(SCENES / f"{name}.npz"))
    w = dict(np.load(SCENES / f"{name}_wm.npz"))
    s["wm_latents"], s["z0"] = w["latents"].astype(np.float32), w["z0"].astype(np.float32)
    return s


def composite(s, cond, t):
    if f"{cond}_mean_stay" in s and f"{cond}_mean_topple" in s:
        return (0.5 * s[f"{cond}_mean_stay"][t].astype(np.float32) + 0.5 * s[f"{cond}_mean_topple"][t]).astype(np.uint8)
    return s[f"{cond}_mean_all"][t]


def projections(s):
    """Project every engine's latent futures (minus the start latent) on the real fork direction."""
    real = s["truth_A_latents"].astype(np.float32).reshape(64, 19, -1)
    end = real[:, 18] - real[:, 18].mean(0)
    u = np.linalg.svd(end, full_matrices=False)[2][0]
    out = {}
    for key, lat in (("real", real), ("probe", s["aware_latents"].astype(np.float32).reshape(64, 19, -1)),
                     ("wm", s["wm_latents"].reshape(64, 19, -1))):
        out[key] = (lat - s["z0"].reshape(1, 1, -1)) @ u
    top = s["truth_A_topples"]
    if top.any() and (~top).any() and out["real"][top, 18].mean() < out["real"][~top, 18].mean():
        out = {k: -v for k, v in out.items()}
    return out


def disagreement(lat):
    """(64, 28, 28, 8) -> (28, 28) spread of the 64 futures per location."""
    return np.sqrt(((lat - lat.mean(0)) ** 2).sum(-1).mean(0))


def verdict(alarm, fork):
    if alarm and fork:
        return "ALARM", "caught the fork", GREEN
    if alarm:
        return "ALARM", "false alarm", RED
    if fork:
        return "quiet", "missed the fork", RED
    return "quiet", "correctly quiet", GREEN


# ----------------------------------------------------------------------------- scenario frames
def scenario_frame(s, row, name, idx, t, show_verdict, scale):
    fig = new_fig()
    title, caption = CAPTIONS[name]
    text(fig, 40, 22, f"Scenario {idx} of 5:  {title}", 34, weight="bold")
    text(fig, 40, 70, f"set C (held-out test) state ep{row['episode']} step {row['index']}   |   "
                      f"control step {t + 1} of 38" + ("   (chunk)" if t < 8 else "   (30-step hold, imagined by the monitor)"),
         17, MUTED)
    cols = [40, 530, 1020]
    heads = [("Reality (64 true futures)", "oracle, from the true scene"),
             ("MuJoCo probe (64 imagined)", "from the image + sampled errors"),
             ("World model (64 imagined)", "learned, deterministic; latent only")]
    for x, (h1, h2) in zip(cols, heads):
        text(fig, x, 112, h1, 19, weight="bold")
        text(fig, x, 140, h2, 13.5, MUTED)
    for x, cond in zip(cols[:2], ("truth_A", "aware")):
        a = ax_px(fig, x, 165, 470, 470); a.imshow(composite(s, cond, t)); a.set_xticks([]); a.set_yticks([])
        k = int(s[f"{cond}_topples"].sum())
        a.text(0.02, 0.97, f"{k}/64 topple a neighbour" if t == 37 else "", transform=a.transAxes, va="top",
               fontsize=17, color="white", weight="bold", bbox=dict(fc=ORANGE if k else TEAL, ec="none", pad=5))
        if f"{cond}_mean_stay" in s and f"{cond}_mean_topple" in s:
            a.text(0.02, 0.03, "both outcome groups overlaid at 50%", transform=a.transAxes, fontsize=12,
                   color="white", bbox=dict(fc="#00000080", ec="none", pad=3))
    # world-model panel: start image + where its futures disagree
    step = max([i for i, f in enumerate(WM_FRAMES) if f <= t], default=None)
    a = ax_px(fig, cols[2], 165, 470, 470)
    gray = s["close_frame"].mean(-1)
    a.imshow(gray, cmap="gray", vmin=0, vmax=420, extent=(0, 28, 28, 0))
    if step is not None:
        d = disagreement(s["wm_latents"][:, step])
        a.imshow(d / scale, cmap="magma", vmin=0, vmax=1, alpha=np.clip(d / scale, 0, 1) * 0.85,
                 extent=(0, 28, 28, 0), interpolation="bilinear")
        ia = a.inset_axes([0.64, 0.64, 0.35, 0.35])
        ia.imshow(gray, cmap="gray", vmin=0, vmax=420, extent=(0, 28, 28, 0))
        dr = disagreement(s["truth_A_latents"][:, step].astype(np.float32))
        ia.imshow(dr / scale, cmap="magma", vmin=0, vmax=1, alpha=np.clip(dr / scale, 0, 1) * 0.85,
                  extent=(0, 28, 28, 0), interpolation="bilinear")
        ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_edgecolor("white"); sp.set_linewidth(2)
        ia.text(0.04, 0.04, "real futures", transform=ia.transAxes, fontsize=11, color="white",
                bbox=dict(fc="#00000080", ec="none", pad=2))
    a.set_xticks([]); a.set_yticks([])
    a.text(0.02, 0.03, "heat = where its 64 futures disagree", transform=a.transAxes, fontsize=12,
           color="white", bbox=dict(fc="#00000080", ec="none", pad=3))
    # fans
    proj = projections(s)
    allv = np.concatenate([v.ravel() for v in proj.values()])
    lo, hi = min(allv.min(), -1.0), max(allv.max(), 1.0)
    pad = 0.08 * (hi - lo); lo, hi = lo - pad, hi + pad
    for x, key, label in zip(cols, ("real", "probe", "wm"), ("real", "MuJoCo probe", "world model")):
        f = ax_px(fig, x + 55, 690, 415, 300)
        f.set_xlim(0, 18.5); f.set_ylim(lo, hi)
        f.axvspan(3.5, 18.5, color=CARD, zorder=0)
        for hold in (8, 18):
            f.axvline(hold, color=LINE, lw=1.2, ls="--", zorder=1)
        if step is not None:
            xs = np.arange(step + 1)
            for j in range(64):
                if key == "wm":
                    c = VIOLET
                else:
                    c = ORANGE if s["truth_A_topples" if key == "real" else "aware_topples"][j] else TEAL
                f.plot(xs, proj[key][j, :step + 1], color=c, lw=1.1, alpha=0.55)
        f.set_xticks([0, 4, 8, 18]); f.set_xticklabels(["start", "hold", "hold 10", "hold 30"], fontsize=11)
        f.tick_params(axis="y", labelsize=10)
        f.set_title(f"{label}: projection on the real fork direction", fontsize=13, color=MUTED, loc="left")
        for sp in ("top", "right"):
            f.spines[sp].set_visible(False)
    text(fig, 40, 1030, "teal = stays,  orange = topples,  violet = world-model future (its outcome is not observable)   |   "
                        "split test reads the futures at hold 10 and hold 30 (dashed)", 13, MUTED)
    # verdict column
    card(fig, 1520, 112, 360, 878)
    text(fig, 1545, 132, "The monitor's decision", 20, weight="bold")
    fork = row["fork"]
    k = row["truth_topples"]
    text(fig, 1545, 180, "Truth", 16, MUTED)
    text(fig, 1545, 205, ("FORK" if fork else "SAFE") if show_verdict else "...",
         30, (ORANGE if fork else TEAL) if show_verdict else MUTED, weight="bold")
    text(fig, 1545, 250, f"{k}/64 real futures topple" if show_verdict else "", 14, MUTED)
    text(fig, 1545, 272, "fork = 2-62 of 64 topple" if show_verdict else "", 12, MUTED)
    for y, label, alarm in ((330, "MuJoCo probe", row["probe_alarm"]), (520, "World model", row["wm_alarm"])):
        text(fig, 1545, y, label, 16, MUTED)
        if show_verdict:
            dec, outcome, col = verdict(alarm, fork)
            text(fig, 1545, y + 25, dec, 30, weight="bold")
            text(fig, 1545, y + 72, ("✓ " if col == GREEN else "✗ ") + outcome, 20, col, weight="bold")
        else:
            text(fig, 1545, y + 25, "imagining ...", 22, MUTED)
    if show_verdict:
        text(fig, 1545, 700, wrap(caption, 30), 14.5, INK, linespacing=1.45)
    return grab(fig)


# ----------------------------------------------------------------------------- slides
def slide_title():
    fig = new_fig()
    text(fig, W / 2, 330, "Accurate on Average, Blind at the Fork", 58, weight="bold", ha="center")
    text(fig, W / 2, 430, "Can a robot see that its next move might knock something over?", 30, MUTED, ha="center")
    text(fig, W / 2, 520, "A fork monitor in the spirit of an intuitive physics engine: propagate the robot's own measured "
                          "uncertainty through a rough simulator,\nand flag the action chunk if the imagined futures split.",
         20, INK, ha="center", linespacing=1.5)
    text(fig, W / 2, 660, "MuJoCo (privileged physics)   vs   a learned deterministic world model   |   simulation results",
         18, MUTED, ha="center")
    return grab(fig)


def slide_task(s):
    fig = new_fig()
    text(fig, 80, 60, "The task", 40, weight="bold")
    a = ax_px(fig, 80, 150, 620, 620); a.imshow(s["close_frame"]); a.set_xticks([]); a.set_yticks([])
    text(fig, 80, 790, "the robot's close camera (one image per decision)", 15, MUTED)
    lines = [("Three Jenga-like blocks stand upright.", INK),
             ("The robot must pick the middle red block\nwithout toppling its neighbours.", INK),
             ("", INK),
             ("Before each 8-step action chunk, the monitor asks:", MUTED),
             ("Could this chunk end two different ways,\nsome futures staying up and some toppling a neighbour?", INK),
             ("", INK),
             ("If yes, the state is a fork: small, unmeasurable errors\ndecide the outcome, so the chunk is fragile.", ORANGE)]
    y = 170
    for s_, c in lines:
        text(fig, 780, y, s_, 26 if c != MUTED else 22, c, linespacing=1.35)
        y += 46 * (s_.count("\n") + 1) if s_ else 30
    for x, key, label, col in ((780, "truth_A_one_stay", "one future: all stay", TEAL),
                               (1130, "truth_A_one_topple", "another: one topples", ORANGE)):
        a = ax_px(fig, x, 640, 300, 300); a.imshow(s[key][37]); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(4)
        text(fig, x, 950, label, 16, col, weight="bold")
    text(fig, 1450, 700, wrap("Same scene, same chunk; only tiny perception and execution errors differ.", 24), 16, MUTED,
         linespacing=1.5)
    return grab(fig)


def slide_monitor(sc):
    fig = new_fig()
    text(fig, 80, 50, "The fork monitor", 40, weight="bold")
    boxes = [("Camera image", "one close-camera frame per decision"),
             ("Its own uncertainty", "perception error (measured offline on held-out images) + execution error "
                                     "(measured snippets)"),
             ("Rollout engine", "MuJoCo with privileged shapes, masses and friction, OR a learned world model on "
                                "DINOv2 tokens"),
             ("64 imagined futures", "the 8-step chunk + a 30-step hold that exists only in the monitor's imagination"),
             ("Split test", "two persistent groups at hold 10 and hold 30?  If yes: alarm")]
    x = 55
    for k, (h1, h2) in enumerate(boxes):
        card(fig, x, 150, 330, 470, color=CARD if k != 2 else "#efe9fb")
        text(fig, x + 18, 170, h1, 21, weight="bold")
        text(fig, x + 18, 215, wrap(h2, 27), 15, MUTED, linespacing=1.45)
        if k < 4:
            text(fig, x + 337, 360, "\u2192", 30, MUTED)
        x += 370
    a = ax_px(fig, 75, 380, 220, 220); a.imshow(sc["close_frame"]); a.axis("off")
    a = ax_px(fig, 1185, 380, 220, 220); a.imshow(composite(sc, "aware", 37)); a.axis("off")
    proj = projections(sc)
    f = ax_px(fig, 1570, 390, 270, 200)
    for j in range(64):
        f.plot(proj["probe"][j], color=ORANGE if sc["aware_topples"][j] else TEAL, lw=1, alpha=0.6)
    for hold in (8, 18):
        f.axvline(hold, color=INK, lw=1, ls="--")
    f.set_xticks([8, 18]); f.set_xticklabels(["hold 10", "hold 30"], fontsize=10); f.set_yticks([])
    for sp in ("top", "right", "left"):
        f.spines[sp].set_visible(False)
    f.patch.set_alpha(0)
    text(fig, 80, 670, "Ground truth (for scoring only)", 24, weight="bold")
    text(fig, 80, 715, wrap("64 futures run from the TRUE scene with the same kinds of error. The state is a fork if 2-62 "
                            "of the 64 topple a neighbour (tilt over 45 deg). A second independent oracle run agrees on 99% "
                            "of states, so the label is repeatable.", 125), 18, INK, linespacing=1.5)
    text(fig, 80, 840, "What to watch in each scenario", 24, weight="bold")
    text(fig, 80, 885, wrap("Top: what really happens (left), what the robot imagines in MuJoCo (middle) and with the learned "
                            "world model (right). Bottom: every future traced over time; a fork shows up as two separating "
                            "bundles.", 125), 18, INK, linespacing=1.5)
    return grab(fig)


def bars(ax, rows, title):
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    caught = [r[1] for r in rows]; false = [r[2] for r in rows]
    ax.barh(y + 0.2, caught, 0.38, color=TEAL, label="forks caught")
    ax.barh(y - 0.2, false, 0.38, color=ORANGE, label="false alarms")
    for yy, c, f in zip(y, caught, false):
        ax.text(c + 1, yy + 0.2, f"{c:.0f}%", va="center", fontsize=14, color=INK)
        ax.text(f + 1, yy - 0.2, f"{f:.0f}%", va="center", fontsize=14, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=15, color=INK)
    ax.set_xlim(0, 112); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.tick_params(axis="x", labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=19, loc="left", color=INK, weight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=13, frameon=False)


def slide_results():
    c = json.loads((ROOT / "results/jenga/fork_label_reanalysis.json").read_text())["set_C_panel4"]
    pct = lambda d: (100 * d["recall"], 100 * d["false_alarm"])
    rows = [("oracle, second run (ceiling)", *pct(c["ceiling_B_vs_A"])),
            ("MuJoCo probe from images", *pct(c["probe_images"])),
            ("MuJoCo probe, 5 pose hypotheses", *pct(c["probe_5hyp_images"])),
            ("MuJoCo, perception error ignored", *pct(c["naive"])),
            ("deterministic world model", *pct(c["wm_deterministic"])),
            ("stochastic world model", *pct(c["wm_stochastic"]))]
    fig = new_fig()
    text(fig, 80, 50, "Results on set C (held-out test set)", 40, weight="bold")
    text(fig, 80, 110, f"{c['states']} states, {c['forks_A']} forks", 20, MUTED)
    ax = ax_px(fig, 520, 190, 1250, 700)
    bars(ax, rows, "")
    text(fig, 80, 935, wrap("Physics with the robot's own uncertainty catches most forks from one image. The learned "
                            "deterministic world model, accurate on average, misses about 4 in 10; the stochastic one, whose "
                            "noise should represent uncertainty, catches fewer than 1 in 5.", 150), 18, INK, linespacing=1.5)
    return grab(fig)


def slide_learned():
    def get(f, k=None):
        d = json.loads((ROOT / f).read_text())
        d = d[k] if k else d
        return 100 * d["caught"], 100 * d["false_alarm"]
    rows = [("iteration-2 recipe (uncertain starts)", *get("results/jenga/phase1_iter2_ref_on_H.json")),
            ("E: 5x fork-rich data", *get("results/jenga/phase1_selection.json", "E")),
            ("E + 3x training", *get("results/jenga/phase1b_selection.json", "a1")),
            ("E + 3x training + 9x fork data", *get("results/jenga/phase1b_selection.json", "a2")),
            ("E + 3-frame history", *get("results/jenga/phase1c_selection.json", "conv3"))]
    gen = json.loads((ROOT / "results/jenga/phase1_selection.json").read_text())
    best_gen = max(100 * gen[k]["caught"] for k in "ABCDF")
    fig = new_fig()
    text(fig, 80, 50, "Can a learned world model close the gap?", 40, weight="bold")
    text(fig, 80, 110, "held-out selection set H (160 states, 75 forks; a different set from set C, so not directly comparable)",
         20, MUTED)
    ax = ax_px(fig, 520, 190, 1250, 620)
    bars(ax, rows, "")
    ax.axvline(80, color=INK, lw=1.5, ls="--")
    ax.text(80.5, len(rows) - 0.35, "gate: 80% caught\n(with at most 20% false alarms)", fontsize=12, color=INK, va="top")
    text(fig, 80, 860, f"Generative variants (flow matching, energy score, whole-clip): at most {best_gen:.0f}% of forks caught.",
         18, INK)
    text(fig, 80, 905, wrap("Giving the deterministic model uncertain starts helps a lot, but every variant so far plateaus "
                            "near 70%. Still running: DINO-WM's transformer predictor. Next: proprioception input and a "
                            "fine-tuned pretrained V-JEPA 2-AC.", 150), 18, INK, linespacing=1.5)
    return grab(fig)


def slide_takeaways():
    fig = new_fig()
    text(fig, 80, 80, "Takeaways for safe deployment", 40, weight="bold")
    points = ["Physical uncertainty must be propagated, not just estimated: ignoring perception error drops forks "
              "caught from 88% to 67%.",
              "A world model can be accurate on average and still blind at the fork: one bundle of futures where "
              "reality splits in two.",
              "Benchmarks for world models used in safety monitors should test whether their futures branch where the "
              "physics branches.",
              "Until learned dynamics can do that, the intuitive-physics design wins: learned perception and measured "
              "uncertainty feeding an approximate physics engine."]
    y = 200
    for k, p in enumerate(points, 1):
        text(fig, 80, y, f"{k}", 34, TEAL, weight="bold")
        text(fig, 140, y + 4, wrap(p, 95), 23, INK, linespacing=1.5)
        y += 160
    text(fig, 80, 960, "Simulation only; the MuJoCo engine uses the simulator's true shapes, masses and friction.", 16, MUTED)
    return grab(fig)


# ----------------------------------------------------------------------------- main
def main():
    import imageio_ffmpeg
    scenarios = json.loads((SCENES / "scenarios.json").read_text())
    scenes = {n: load_scene(n) for n in scenarios}
    scale = np.percentile(np.concatenate([disagreement(s["truth_A_latents"][:, 18].astype(np.float32)).ravel()
                                          for s in scenes.values()]), 99)
    writer = imageio_ffmpeg.write_frames(str(OUT), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
                                         output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)

    def put(img, seconds):
        for _ in range(int(round(seconds * FPS))):
            writer.send(img.tobytes())
    put(slide_title(), 5)
    put(slide_task(scenes["2_fork_both"]), 9)
    put(slide_monitor(scenes["2_fork_both"]), 14)
    for idx, (name, row) in enumerate(scenarios.items(), 1):
        s = scenes[name]
        put(scenario_frame(s, row, name, idx, 0, False, scale), 2.5)
        for t in range(38):
            put(scenario_frame(s, row, name, idx, t, False, scale), 7 / FPS)
        put(scenario_frame(s, row, name, idx, 37, True, scale), 7)
        print(name, "done", flush=True)
    put(slide_results(), 11)
    put(slide_learned(), 12)
    put(slide_takeaways(), 10)
    writer.close()
    print(OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        scenarios = json.loads((SCENES / "scenarios.json").read_text())
        name = sys.argv[2]
        s = load_scene(name)
        scale = np.percentile(disagreement(s["truth_A_latents"][:, 18].astype(np.float32)), 99)
        from PIL import Image
        idx = list(scenarios).index(name) + 1
        Image.fromarray(scenario_frame(s, scenarios[name], name, idx, 37, True, scale)).save(HERE / f"preview_{name}.png")
        for f, img in (("title", slide_title()), ("monitor", slide_monitor(load_scene("2_fork_both"))), ("results", slide_results()),
                       ("learned", slide_learned()), ("takeaways", slide_takeaways()), ("task", slide_task(s))):
            Image.fromarray(img).save(HERE / f"preview_{f}.png")
    else:
        main()
