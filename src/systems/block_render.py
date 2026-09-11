"""
Rasterise the tipping block to a 224x224 RGB frame — Phase A of PLAN_DINOWM.md.

The point of this module is to stand in for a camera looking at the block, so that the world model
sees PIXELS rather than (theta, omega). That change is not cosmetic: a single frame shows theta but
NOT omega, so the observation stops being Markov and history becomes mandatory. That is the whole
reason the image swap is worth doing.

THE NUISANCE IS DELIBERATE. A black polygon on a white field would give DINOv2 almost nothing to
encode, and a monitor that worked on it would prove very little about a cluttered tabletop. So the
scene carries, on purpose:

  * a textured wall and a perspective-striped tabletop  — so patch features are not degenerate
  * a projected, blurred drop shadow                    — see below
  * per-episode lighting variation                      — shadow direction/length and ambient move

The shadow matters most. Shadow artifacts are the documented precision bottleneck of the real Jenga
monitor (CLAUDE.md, Limitations 2): the world model predicts them inconsistently between the
original and perturbed rollouts, which inflates d_end and produces false positives. Rendering them
here puts that exact failure mode inside a system whose ground truth is computable, which is
something the Jenga setup can never offer.

GEOMETRY IS ORTHOGRAPHIC AND EXACT. The block is drawn side-on with no perspective, so theta is
recoverable from the silhouette without distortion. Only the tabletop and the shadow are faked into
pseudo-3D; they are scenery, never the signal.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import tipping_block as tb

ALPHA = tb.ALPHA_DEFAULT
SIZE = 224                       # DINOv2 native
SS = 2                           # supersampling factor; edges alias badly at 1x

# World window, square so the scale is uniform in x and y (no stretch).
# Block extent over all theta is x in [-2.22, 2.22], y in [0, 2.0]; see PLAN_DINOWM Phase A.
# The fallen block reaches x = +-2.22, but framing for it would shrink the standing block to ~38%
# of the frame height (about 6 patches). Crop tighter and let the fallen block run off the edge:
# "lying flat toward the left" stays unmistakable even clipped, and the standing block gets 57%.
X0, X1 = -1.75, 1.75
Y0, Y1 = -1.10, 2.40             # ground line ~31% up the frame, block top ~88%

_SQUASH = 0.50                   # how flat the ground plane looks (pseudo-3D)


def corners(theta, alpha=ALPHA):
    """Block outline in world coordinates. Pivots on whichever corner it is rocking about."""
    a, b = np.sin(alpha), np.cos(alpha)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (base - piv) @ np.array([[c, -s], [s, c]]).T + piv


def _to_px(pts, size):
    """World -> pixel. y is flipped; scale is shared so circles stay circles."""
    k = size / (X1 - X0)
    x = (pts[:, 0] - X0) * k
    y = size - (pts[:, 1] - Y0) * k
    return np.stack([x, y], 1)


def sample_lighting(rng):
    """Per-episode scene nuisance. Everything here is invisible to the dynamics."""
    return dict(
        skew=float(rng.uniform(-0.85, 0.85)),        # light azimuth -> shadow direction/length
        shadow=float(rng.uniform(0.38, 0.70)),       # shadow opacity
        blur=float(rng.uniform(2.5, 6.0)),           # penumbra softness
        ambient=float(rng.uniform(0.82, 1.15)),      # overall brightness
        warmth=float(rng.uniform(-0.05, 0.08)),      # colour temperature drift
        wall=float(rng.uniform(0.60, 0.76)),         # wall value
        block=float(rng.uniform(0.20, 0.34)),        # block value
    )


def _background(size, light, seed=0):
    """Wall above the ground line, perspective-striped tabletop below. Static per episode."""
    k = size / (X1 - X0)
    horizon = int(round(size - (0.0 - Y0) * k))      # pixel row of the ground line
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.float32)

    # --- wall: vertical gradient + low-frequency mottling so patches are not flat
    rows = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    wall = light["wall"] * (0.88 + 0.20 * rows)
    coarse = rng.normal(0, 1, (max(horizon // 16, 2), max(size // 16, 2))).astype(np.float32)
    coarse = np.array(Image.fromarray(coarse).resize((size, horizon), Image.BICUBIC))
    img[:horizon] = (wall + 0.022 * coarse)[..., None]

    # --- tabletop: stripes that compress toward the horizon, faking a receding plane
    d = np.arange(size - horizon, dtype=np.float32) + 1.0          # depth below the line
    world_d = d / (k * _SQUASH)                                    # -> world units toward viewer
    stripe = 0.5 + 0.5 * np.sin(world_d * 7.5)
    table = 0.34 + 0.150 * stripe[:, None]
    grain = rng.normal(0, 1, (max((size - horizon) // 8, 2), max(size // 8, 2))).astype(np.float32)
    grain = np.array(Image.fromarray(grain).resize((size, size - horizon), Image.BICUBIC))
    img[horizon:] = (table + 0.030 * grain)[..., None]

    img[..., 0] *= 1.0 + light["warmth"]                           # warm/cool drift
    img[..., 2] *= 1.0 - light["warmth"]
    return img, horizon


def _shadow_mask(theta, size, light, alpha=ALPHA):
    """Project the block onto the ground plane along the light direction, then blur.

    A world point at height y lands at x + y*skew, squashed to -y*SQUASH so it falls inside the
    visible tabletop band rather than on the (edge-on) ground line."""
    C = corners(theta, alpha)
    proj = np.stack([C[:, 0] + C[:, 1] * light["skew"], -C[:, 1] * _SQUASH], 1)
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in _to_px(proj, size)], fill=255)
    return m.filter(ImageFilter.GaussianBlur(light["blur"] * size / SIZE))


def render(theta, light, size=SIZE, ss=SS, bg=None, alpha=ALPHA):
    """One frame. Pass `bg` (from `prepare`) to avoid rebuilding the static scene every step."""
    S = size * ss
    if bg is None:
        bg = _background(S, light)[0]
    img = bg.copy()

    sh = np.asarray(_shadow_mask(theta, S, light, alpha), np.float32) / 255.0
    img *= (1.0 - light["shadow"] * sh)[..., None]

    C = _to_px(corners(theta, alpha), S)
    face = Image.new("L", (S, S), 0)
    ImageDraw.Draw(face).polygon([tuple(p) for p in C], fill=255)
    face = np.asarray(face, np.float32) / 255.0
    # a touch of shading on the block so its two visible faces are not one flat value
    shade = light["block"] * (0.86 + 0.28 * np.linspace(0, 1, S, dtype=np.float32)[None, :])
    img = img * (1.0 - face[..., None]) + shade[..., None] * face[..., None]

    img = np.clip(img * light["ambient"], 0.0, 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8))
    if ss != 1:
        out = out.resize((size, size), Image.LANCZOS)
    return np.asarray(out)


def prepare(light, size=SIZE, ss=SS, seed=0):
    """Build the static background once per episode; reuse across all its frames."""
    return _background(size * ss, light, seed)[0]


def render_traj(thetas, light, size=SIZE, ss=SS, seed=0, alpha=ALPHA):
    """(T,) angles -> (T, size, size, 3) uint8."""
    bg = prepare(light, size, ss, seed)
    return np.stack([render(t, light, size, ss, bg, alpha) for t in thetas])
