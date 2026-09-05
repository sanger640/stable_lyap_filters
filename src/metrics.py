"""
Divergence metrics over DINO world-model latents, plus the patch masks that matter.

These are the *detection* metrics: given an action chunk and a short observation history,
score how unstable the predicted future is. Everything here is consolidated from the wider
research codebase; only the variants that survived held-out validation are kept.

Ranking established on 1772 chunks / 25 unsafe (100 episodes, jenga_noise_50), AUC:

    ftle (original: max over patches AND perturbations)      0.599   <- near chance
    ftle, max patch / mean perturbation                      0.685
    d_end, mean                                              0.780
    d_end, p90 over patches                                  0.799
    d_end + low-norm patch mask (k=30)                       0.854
    d_end + PC1 background mask (75% keep)                   0.894
    ftle_variance + PC1 background mask                      0.896

The headline lesson of the detection work: the FTLE *ratio* is the problem. d_start is
measured one prediction step in, so it is tiny and noisy, and dividing by it reorders patches
by how quiet they happened to start rather than by how unstable they are. Dropping the
denominator entirely (`d_end`) is worth ~0.2 AUC. A second maximum (over perturbations) makes
it worse still: an extremum over ~4100 values per chunk tracks tail noise.
"""
import numpy as np
import torch

PATCH_GRID = 14
NUM_PATCHES = PATCH_GRID * PATCH_GRID          # 196: dinov2_vits14 on a 196px encoder input
MASKED_ROWS = (0, 1, 8, 9, 10, 11, 12, 13)     # ceiling/upper background + checkered floor


def build_patch_keep_mask(num_patches=NUM_PATCHES, device="cpu"):
    """Geometric row mask. In the fixed external view the task occupies only a horizontal
    band (rows 2-7). Rows 0-1 are ceiling/background and rows 8-13 are the checkered floor --
    the floor is the highest-texture region in frame, so before masking it competed for (and
    routinely won) the max in every chunk."""
    keep = torch.ones(num_patches, dtype=torch.bool, device=device)
    for r in MASKED_ROWS:
        keep[r * PATCH_GRID:(r + 1) * PATCH_GRID] = False
    return keep


def low_norm_mask(z_obs, keep, k=30):
    """Drop the k lowest-||z|| patches among those already kept.

    Counterintuitive but validated: divergence concentrates on LOW-norm patches, not the
    high-norm register/artifact tokens that were the original suspect. Cosine distance divides
    by ||z||, so a near-featureless patch has a short, poorly-determined direction that wobbles
    under any perturbation. corr(||z||, d_end) = -0.641 on ground-truth-static patches.
    k~=30 was selected on one half of the episodes and held on the other (20 splits)."""
    m = keep.clone()
    idx = torch.where(m)[0]
    if k > 0 and idx.numel() > k:
        order = idx[torch.argsort(z_obs[idx].norm(dim=-1))]
        m[order[:k]] = False
    return m


def fit_pc1_basis(z_obs_bank, keep):
    """Fit the PC1 direction used by `pc1_mask`, on a bank of safe-chunk patch embeddings.

    Returns (mu, pc1). NOTE the sign convention is resolved by the caller against ground-truth
    patch motion -- see `pc1_mask`."""
    bank = z_obs_bank[:, keep].reshape(-1, z_obs_bank.shape[-1]).cpu().numpy()
    mu = bank.mean(0)
    _, _, Vt = np.linalg.svd(bank - mu, full_matrices=False)
    return mu, Vt[0]


def pc1_mask(z_obs, keep, mu, pc1, sign=+1, keep_pct=75):
    """Keep the top `keep_pct`% of patches by signed PC1 projection.

    IMPORTANT, and the single most counterintuitive result in this project: the direction that
    works keeps the LOW-MOTION (background) patches, not the moving foreground. This was
    originally mislabelled because the sign was chosen by a norm heuristic ("higher mean ||z||
    = foreground") that was never checked and turned out to be backwards on both datasets
    tested -- corr(PC1_raw, ground-truth patch motion) = -0.479 and -0.486. Always resolve the
    sign against measured patch motion, never against norm.

    Why background wins (validated, not just rationalised): on foreground patches
    corr(motion, d_end) = +0.45 and is just as strong in SAFE chunks (+0.452) as unsafe
    (+0.386) -- i.e. most of their divergence is a motion/phase confound from the always-moving
    arm, not a failure signal. Background patches have ~zero baseline motion, so any divergence
    there actually means something. Sweeping keep_pct 10-100 per sign with held-out selection
    picks 75% for background in 20/20 splits; the foreground direction's own optimum is
    "no filtering at all"."""
    m = keep.clone()
    if keep_pct >= 100:
        return m
    proj = ((z_obs.cpu().numpy() - mu) @ pc1) * sign
    thr = np.percentile(proj[keep.cpu().numpy()], 100 - keep_pct)
    m &= torch.as_tensor(proj >= thr, device=m.device)
    return m


def d_end(z_pred, n_hist, mask=None, reduce="p90"):
    """Divergence of the predicted final latent from the last observed one, per patch.

    This is the "nominal" metric: a single unperturbed rollout, no Deviator Agent. On
    jenga_noise_50 it matched the full 50-perturbation apparatus (0.826 vs 0.851 AUC,
    difference not significant) at 53 ms vs 2021 ms. The 50-rollout version is NOT
    uniformly better -- on a second dataset the ordering reversed -- so treat
    nominal-vs-perturbed as dataset-dependent rather than settled.

    z_pred: (T, P, F) latents from a rollout; index n_hist is the last observed step.
    """
    cos = torch.nn.functional.cosine_similarity
    per_patch = 1 - cos(z_pred[n_hist], z_pred[-1], dim=-1)      # (P,)
    v = per_patch if mask is None else per_patch[mask]
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return float("nan")
    return {"mean": v.mean(), "p90": torch.quantile(v.float(), 0.90),
            "max": v.max()}[reduce].item()


def ftle_variance(z_pert_final, mask=None, reduce="p90"):
    """Spread of the perturbed rollouts' final latents about their own centroid.

    Structurally cleaner than the FTLE ratio -- there is no denominator, and it never
    references the unperturbed trajectory, so world-model error on the nominal rollout drops
    out. Performs on par with d_end (0.896 vs 0.894 with the PC1 mask). It does NOT escape the
    low-||z|| noise mechanism though: corr(||z||, ftle_variance) = -0.646, essentially
    identical to d_end's -0.641.

    z_pert_final: (N_perturbations, P, F).
    """
    cos = torch.nn.functional.cosine_similarity
    centroid = z_pert_final.mean(0, keepdim=True)
    per_patch = (1 - cos(z_pert_final, centroid.expand_as(z_pert_final), dim=-1)).mean(0)
    v = per_patch if mask is None else per_patch[mask]
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return float("nan")
    return {"mean": v.mean(), "p90": torch.quantile(v.float(), 0.90),
            "max": v.max()}[reduce].item()


def ftle_ratio(z_orig, z_pert, n_hist, mask=None):
    """The ORIGINAL metric: (1/T) log(d_end / d_start), max over patches then perturbations.

    Kept only for reproducibility of the baseline. AUC 0.599 -- near chance. Do not use it;
    see the module docstring for why the ratio and the double maximum both hurt."""
    cos = torch.nn.functional.cosine_similarity
    d = 1 - cos(z_pert, z_orig.expand_as(z_pert), dim=-1)        # (N, T, P)
    ds, de = d[:, n_hist] + 1e-4, d[:, -1] + 1e-4
    T = z_pert.shape[1] - n_hist
    lam = (1.0 / T) * torch.log(de / ds)
    lam = torch.where(de > 1e-3, lam, torch.full_like(lam, -float("inf")))
    if mask is not None:
        lam[:, ~mask] = -float("inf")
    finite = lam[torch.isfinite(lam)]
    return float(finite.max()) if finite.numel() else float("nan")


def auc(pos, neg):
    """Rank-based AUC, ties counted at half weight."""
    p = np.asarray(pos, float); n = np.asarray(neg, float)
    p = p[np.isfinite(p)]; n = np.sort(n[np.isfinite(n)])
    if not len(p) or not len(n):
        return float("nan")
    r = (np.searchsorted(n, p, "left") +
         0.5 * (np.searchsorted(n, p, "right") - np.searchsorted(n, p, "left")))
    return float(r.mean() / len(n))
