"""Does LIGHTING dominate POSE in the latent geometry the monitor clusters on?

The monitor assigns basins by `argmin_c ||E - C_c||^2` -- nearest centroid, Euclidean. That is a
pure distance operation, so it only works if physical state separates latents MORE than the
lighting nuisance does. If the same fallen block under two lamps is further apart than fall-left is
from fall-right under one lamp, clustering sorts by lamp and Phases E-F measure scenery.

Phase C cannot answer this: it measures whether a PREDICTOR can decode pose despite lighting, which
is a different question from whether the SPACE is metrically organised by pose. And the test is
fair without a trained model, because VWorldModel.predict() maps patch tokens to patch tokens --
the predictor's outputs live in the same space as the encoder's.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import block_render as br                                          # noqa: E402
from phase_b_probe import encode                                   # noqa: E402

POSES = [-np.pi / 2, -0.7, -0.35, 0.0, 0.35, 0.7, np.pi / 2]
NAMES = ["fallL", "-0.70", "-0.35", "up", "+0.35", "+0.70", "fallR"]
TERM = [0, 3, 6]                                        # the three attractors
N_LIGHT = 40

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()

lights = [br.sample_lighting(np.random.default_rng(1000 + i)) for i in range(N_LIGHT)]
frames = np.stack([br.render(p, L, bg=br.prepare(L, seed=i))
                   for i, L in enumerate(lights) for p in POSES])
Z = encode(frames, model, dev).reshape(len(frames), -1)             # (N_LIGHT*P, 256*384)
Z = Z.reshape(N_LIGHT, len(POSES), -1)                              # (light, pose, D)
print(f"encoded {len(frames)} frames -> {Z.shape}\n")


def scatter(sub, names):
    """between-pose vs within-pose(across-lighting) scatter."""
    cen = sub.mean(0)                                               # (pose, D)
    within = np.sqrt(((sub - cen[None]) ** 2).sum(-1)).mean()       # same pose, diff lighting
    P = len(names)
    dists = [np.linalg.norm(cen[i] - cen[j]) for i in range(P) for j in range(i + 1, P)]
    between = float(np.mean(dists))
    return within, between, min(dists)


for tag, idx in (("THREE TERMINAL STATES (what the monitor clusters)", TERM),
                 ("ALL SEVEN POSES", list(range(len(POSES))))):
    sub = Z[:, idx]
    nm = [NAMES[i] for i in idx]
    w, b, bmin = scatter(sub, nm)
    print(f"=== {tag} ===")
    print(f"  within-pose  (same pose, 40 lightings)  mean dist {w:8.2f}")
    print(f"  between-pose (centroids)                mean dist {b:8.2f}   min {bmin:8.2f}")
    print(f"  ratio between/within = {b/w:.2f}   (>1 pose dominates, <1 lighting dominates)")
    print(f"  worst-case ratio min-between/within = {bmin/w:.2f}\n")

print("=== OPERATIONAL TEST: nearest centroid, centroids from SEEN lighting only ===")
for idx, tag in ((TERM, "3 terminals"), (list(range(len(POSES))), "7 poses")):
    sub = Z[:, idx]
    for n_fit in (2, 5, 10, 20):
        C = sub[:n_fit].mean(0)                                     # centroids from n_fit lightings
        held = sub[n_fit:]                                          # unseen lighting
        d = ((held[:, :, None] - C[None, None]) ** 2).sum(-1)       # (light, pose, centroid)
        pred = d.argmin(-1)
        acc = (pred == np.arange(len(idx))[None]).mean()
        print(f"  {tag:<12} centroids from {n_fit:>2} lightings -> "
              f"accuracy on {len(held)} unseen lightings: {acc:6.1%}")
    print()
