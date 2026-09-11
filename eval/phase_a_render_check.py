"""Phase A acceptance check for the block renderer (PLAN_DINOWM.md).

Three things have to hold before Phase B is worth starting:

  1. +theta and -theta render VISIBLY DIFFERENTLY. The block pivots on the opposite corner, so the
     silhouettes genuinely differ -- but if the render lost that, the three attractors would
     collapse to two and every downstream result would be void.
  2. The three terminal states (-pi/2, 0, +pi/2) are mutually distinct.
  3. Throughput is enough to render the corpus in ~20 min.
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "systems"))
import block_render as br                                          # noqa: E402

OUT = ROOT / "results" / "phase_a"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
L = br.sample_lighting(rng)


def sheet(frames, labels, path, pad=4, scale=1):
    h, w, _ = frames[0].shape
    W = len(frames) * (w + pad) + pad
    out = np.full((h + 2 * pad, W, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        x = pad + i * (w + pad)
        out[pad:pad + h, x:x + w] = f
    im = Image.fromarray(out)
    if scale != 1:
        im = im.resize((W * scale, (h + 2 * pad) * scale), Image.NEAREST)
    im.save(path)
    print(f"  -> {path.name}   [{'  '.join(labels)}]")


print("rendering sweeps ...")
bg = br.prepare(L)

# 1. theta sweep across the whole range, including past the topple point
ths = [-np.pi / 2, -0.7, -0.35, -0.15, 0.0, 0.15, 0.35, 0.7, np.pi / 2]
sheet([br.render(t, L, bg=bg) for t in ths], [f"{t:+.2f}" for t in ths],
      OUT / "sweep.png")

# 2. +/- pairs, the criterion that matters most
pairs = [0.10, 0.20, 0.35, 0.60, 1.20]
sheet([br.render(s * t, L, bg=bg) for t in pairs for s in (+1, -1)],
      [f"{s*t:+.2f}" for t in pairs for s in (+1, -1)], OUT / "sign_pairs.png")

# 3. the three terminal states
term = [-np.pi / 2, 0.0, np.pi / 2]
sheet([br.render(t, L, bg=bg) for t in term], ["fall L", "upright", "fall R"],
      OUT / "terminals.png", scale=2)

# 4. lighting variation at a fixed pose -- the nuisance the model must learn to ignore
sheet([br.render(0.25, br.sample_lighting(np.random.default_rng(s))) for s in range(6)],
      [f"light {s}" for s in range(6)], OUT / "lighting.png")

print("\n=== 1. sign asymmetry: does +theta differ from -theta? ===")
scale = float(np.abs(br.render(0.0, L, bg=bg).astype(float) -
                     br.render(0.35, L, bg=bg).astype(float)).mean())
print(f"{'theta':>8}{'mean|diff|':>12}{'vs 0-to-0.35 ref':>18}")
ok_sign = True
for t in pairs:
    d = float(np.abs(br.render(t, L, bg=bg).astype(float) -
                     br.render(-t, L, bg=bg).astype(float)).mean())
    print(f"{t:>8.2f}{d:>12.2f}{d / scale:>18.2f}")
    if d < 2.0:
        ok_sign = False
print(f"  PASS: every +/- pair differs by >2 grey levels" if ok_sign else "  FAIL")

print("\n=== 2. terminal states mutually distinct ===")
T = [br.render(t, L, bg=bg).astype(float) for t in term]
names = ["fallL", "upright", "fallR"]
ok_term = True
for i in range(3):
    for j in range(i + 1, 3):
        d = float(np.abs(T[i] - T[j]).mean())
        print(f"  {names[i]:>8} vs {names[j]:<8} mean|diff| {d:6.2f}")
        if d < 5.0:
            ok_term = False
print("  PASS" if ok_term else "  FAIL")

print("\n=== 3. monotonicity: |theta| vs distance from upright ===")
z = br.render(0.0, L, bg=bg).astype(float)
ds = [float(np.abs(br.render(t, L, bg=bg).astype(float) - z).mean())
      for t in (0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5)]
print("  " + "  ".join(f"{d:.1f}" for d in ds))
ok_mono = all(b > a for a, b in zip(ds, ds[1:]))
print(f"  monotone increasing: {ok_mono}")

print("\n=== 4. throughput ===")
N = 300
t0 = time.time(); [br.render(0.1 + 0.001 * i, L, bg=bg) for i in range(N)]
per = (time.time() - t0) / N
print(f"  {per*1000:.2f} ms/frame  ({1/per:.0f} fps, supersample {br.SS}x)")
for T_ in (45, 90, 150, 450):
    n = 600 * T_
    print(f"    600 traj x {T_:>3} steps = {n:>7,} frames -> {n*per/60:6.1f} min")

print(f"\nPhase A: {'PASS' if (ok_sign and ok_term and ok_mono) else 'FAIL'}")
