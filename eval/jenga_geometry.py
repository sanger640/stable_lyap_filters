"""JENGA GO/NO-GO, step 1: do toppled and intact scenes separate in DINOv2 space?

No world model involved -- just the frozen encoder on REAL final frames. This is the cheapest test
that can kill the whole approach, because the monitor assigns basins by nearest centroid in this
space. If real photographs of a toppled scene do not sit further from intact ones than lighting and
arm pose move things, then nearest-centroid cannot work no matter how good a predictor is.

The toy measured this at 1.58 (pose beats nuisance) and got 100% nearest-centroid on unseen
lighting. Jenga is harder in the way that matters: the distinction is a NEIGHBOUR TIPPED ~15-95
DEGREES rather than a block flat on its face, in a cluttered scene, with a robot arm in frame that
moves differently in every episode.

Labels carry `peak_tilt_deg` against a 45 deg threshold, so this also checks the GRADED version --
does distance in latent space track how far the block actually tipped? That matters more than the
binary split, because the method predicts PROXIMITY, not outcome.
"""
import argparse, io, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
from phase_e_attractors import pdist, plateau                      # noqa: E402

WKSP = Path("/home/sanger/wksp")
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def encode(frames, model, dev, batch=48):
    out = []
    for i in range(0, len(frames), batch):
        x = frames[i:i + batch].astype(np.float32) / 255.0
        x = (x - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        out.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(WKSP / "panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"))
    ap.add_argument("--labels", default=str(WKSP / "panda_express/labels_noise100.json"))
    ap.add_argument("--size", type=int, default=196, help="dino_wm resizes 224 -> 196 (14x14 patches)")
    ap.add_argument("--last-n", type=int, default=1, help="average the final N frames")
    args = ap.parse_args()
    dev = "cuda"

    L = json.load(open(args.labels))
    env = __import__("lmdb").open(args.lmdb, readonly=True, lock=False)
    with env.begin() as t:
        meta = pickle.loads(t.get(b"__metadata__"))["episodes"]
        eps = sorted(meta, key=int)
        frames = []
        for ep in eps:
            ks = meta[ep]["keys"]["cam2"][-args.last_n:]
            ims = []
            for k in ks:
                im = Image.open(io.BytesIO(t.get(k.encode()))).convert("RGB")
                ims.append(np.asarray(im.resize((args.size, args.size), Image.BICUBIC)))
            frames.append(np.mean(ims, 0).astype(np.uint8))
    F = np.stack(frames)
    print(f"{len(F)} episodes, final frame, {F.shape[1]}x{F.shape[2]}\n")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).to(dev).eval()
    E = encode(F, model, dev).reshape(len(F), -1)
    print(f"encoded -> {E.shape[1]} numbers per scene ({E.shape[1]//384} patches)\n")

    fail = np.array([L[e]["outcome"] != "success" for e in eps])
    tilt = np.array([L[e]["peak_tilt_deg"] for e in eps])
    print(f"ground truth: {fail.sum()} toppled / {len(fail)-fail.sum()} intact  "
          f"(threshold 45 deg); peak tilt {tilt.min():.1f}-{tilt.max():.1f} deg\n")

    print(f"{'PCA dim':>8}{'sep ratio':>11}{'NC acc':>9}{'corr w/ tilt':>14}   note")
    print("-" * 62)
    Xc = (E - E.mean(0)).astype(np.float32)
    V = np.linalg.svd(Xc[::2], full_matrices=False)[2]
    for kd in (0, 2, 4, 8, 16, 32):
        X = Xc if kd == 0 else Xc @ V[:kd].T
        D = pdist(X); scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
        iu = np.triu_indices(len(X), 1); pw = D[iu] / scale
        same = fail[iu[0]] == fail[iu[1]]
        sep = np.median(pw[~same]) / np.median(pw[same])
        # leave-one-out nearest centroid
        hit = 0
        for i in range(len(X)):
            m = np.ones(len(X), bool); m[i] = False
            c0 = X[m & ~fail].mean(0); c1 = X[m & fail].mean(0)
            hit += (np.linalg.norm(X[i]-c1) < np.linalg.norm(X[i]-c0)) == fail[i]
        # does distance from the intact centroid track HOW FAR it tipped?
        d_int = np.linalg.norm(X - X[~fail].mean(0), axis=1)
        r = np.corrcoef(d_int, tilt)[0, 1]
        print(f"{kd if kd else 'full':>8}{sep:>11.3f}{hit/len(X):>9.1%}{r:>14.3f}")
    print("\nsep ratio  = distance BETWEEN toppled/intact vs WITHIN a group (toy got 1.58)")
    print("NC acc     = leave-one-out nearest-centroid accuracy (toy got 100%)")
    print(f"corr w/tilt= does latent distance track peak tilt? (base rate {1-fail.mean():.0%})")


if __name__ == "__main__":
    main()
