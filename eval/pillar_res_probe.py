"""Exploratory (docs/PILLAR_TASK.md follow-up): can REAL DINOv2 latents resolve the cube's lateral offset over the pillar?
Ridge probe (5-fold by seed) from the channel-PCA8 latent to the cube-pillar x offset, current camera (fovy 45) vs a
zoomed one (fovy 20, same position). Moments: the diagnostic's B moments with the cube within 3 cm (xy) of the pillar."""
import glob, json, os, pickle, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PILLAR_HALF_W", "0.008")
ROOT = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import pillar_task as P, pillar_wm as W
snaps, ys, seeds = [], [], []
for f in sorted(glob.glob(str(Path.home() / "wksp/pillar_diag/s*.pkl"))):
    D = pickle.loads(Path(f).read_bytes()); snaps += [(D["seed"], m["snap"]) for m in D["moments"]]
sims = {}
for fov in ("45", "20"):
    os.environ["PILLAR_FOVY"] = fov; s = P.PillarSim(); P.reset(s, 5000); sims[fov] = (s, W.Cam(s))
lat = W.Latent(); X = {"45": [], "20": [], "45raw": [], "20raw": []}
from jenga_vjepa_encode import dino_tokens
for seed, snap in snaps:
    s0 = sims["45"][0]; s0.restore(snap); c, p = P.body_pos(s0, "cube"), P.body_pos(s0, "pillar")
    if np.linalg.norm((c - p)[:2]) > 0.03 or c[2] < p[2] + 0.05:     # cube over the pillar, above its top
        continue
    ys.append((c - p)[0] * 1000); seeds.append(seed)
    for fov, (s, cam) in sims.items():
        s.restore(snap); fr, m = cam(); X[fov].append(lat(fr[None], W.patch_keep(m[None]))[0].astype(np.float32).ravel())
        X[fov + "raw"].append((dino_tokens(lat.enc, fr[None], "cuda")[0].reshape(28, 28, -1).cpu().numpy() * W.patch_keep(m[None])[0][..., None]).astype(np.float16).ravel())
y, seeds = np.array(ys), np.array(seeds); print("moments", len(y), "x offset mm sd", y.std().round(2))
folds = np.array([hash(int(s)) % 5 for s in seeds]); out = {"n": len(y), "y_sd_mm": float(y.std())}
for fov in X:
    A = np.stack(X[fov]).astype(np.float32); A = (A - A.mean(0)) / (A.std(0) + 1e-6); best = None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        pred = np.zeros_like(y)
        for k in range(5):
            tr, te = folds != k, folds == k
            K = A[tr] @ A[tr].T; a = np.linalg.solve(K + lam * np.eye(len(K)), y[tr] - y[tr].mean())
            pred[te] = A[te] @ A[tr].T @ a + y[tr].mean()
        r2 = 1 - ((pred - y) ** 2).mean() / y.var(); rmse = np.sqrt(((pred - y) ** 2).mean())
        if best is None or r2 > best[0]: best = (r2, rmse, lam)
    out[f"fovy{fov}"] = {"cv_r2": round(float(best[0]), 3), "rmse_mm": round(float(best[1]), 2), "lambda": best[2]}
    print(fov, out[f"fovy{fov}"], flush=True)
(ROOT / "results/pillar/diag/res_probe.json").write_text(json.dumps(out, indent=1))
