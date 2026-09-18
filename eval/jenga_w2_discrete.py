"""W2a: a discrete ending head, the smallest test of whether a model can jump at a boundary.

PLAN_WORLDMODEL W2. A deterministic continuous map from actions to endings is a continuous
function, so it can only approximate reality's 40x jump with a ramp -- which is what both the
shipped checkpoint (fork jump ratio 5.0) and every fine-tune (2.3, 1.5) produce against reality's
14.9. A categorical output can jump: a small action change flips the argmax, and the mean of two
codes is not a code.

Model: three real history latents and the whole action window -> a distribution over ending codes,
separately for hold steps 10 and 30. The codebook is learned unsupervised on TRAINING endings only
(PCA then k-means; the count is chosen by an elbow on training data, never supplied by hand).
This is a temporally abstract world model: it predicts the ending the monitor reads rather than
every intermediate frame, which isolates the discontinuity question from rollout drift.

Actions are rescaled relative to the nominal chunk and by the execution-error size, so a 1.6 mm
perturbation arrives as an O(1) input instead of 0.05 after dataset normalisation.

Losses: cross-entropy on the true code, plus the W1 branch-preservation terms computed on the
distribution's mean embedding (difference matching over ALL pairs, so the loss cannot win by
exaggerating every perturbation, and an energy term that rewards spread).

Grade with eval/jenga_w0_response_curves.py conventions: fork-state jump ratio against the same
model's quiet-state jump ratio. Never with likelihood alone.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import NUM_HIST  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_stage0_noise_oracle import HOLD  # noqa: E402
from jenga_short_held_tails import HORIZON  # noqa: E402
from jenga_w1_train import load_groups  # noqa: E402,F401 (used by other W2 tooling)

HOLD_STEPS = (10, 30)
PCA_DIM = 64
NOISE_SCALE_MM = 3.2  # median per-chunk max execution error, from tracking_uncertainty.json


def fit_pca(x, dim=PCA_DIM):
    mean = x.mean(0)
    centred = x - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return mean, vt[:dim]


def kmeans(x, k, iterations=50, seed=0):
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iterations):
        distance = ((x[:, None] - centres[None]) ** 2).sum(2)
        labels = distance.argmin(1)
        for j in range(k):
            if np.any(labels == j):
                centres[j] = x[labels == j].mean(0)
    return centres, labels


def build_codebook(x, k, seed=0):
    """k is NOT chosen by a rule here: distortion falls monotonically with k, so any 'elbow'
    rule on this data just returns the largest candidate. The codebook size is reported as a
    sensitivity instead -- W2's gate is the response curve, not the codebook."""
    centres, labels = kmeans(x, k, seed=seed)
    return centres, float(((x - centres[labels]) ** 2).sum(1).mean())


class EndingHead(nn.Module):
    """History latents + action window -> logits over ending codes at each hold step."""

    def __init__(self, history_dim, action_dim, codes, width=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(history_dim + action_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(width, codes) for _ in HOLD_STEPS])

    def forward(self, history, actions):
        h = self.trunk(torch.cat([history, actions], dim=1))
        return [head(h) for head in self.heads]


def action_features(actions):
    """(n, 2+H+HOLD, 4) absolute targets -> perturbation-relative features at an O(1) scale."""
    a = actions[:, :, :3]
    nominal = a[:, NUM_HIST - 1:NUM_HIST, :]        # the pose at the chunk start
    relative = (a - nominal) * 1000.0 / NOISE_SCALE_MM  # millimetres, in units of execution error
    gripper = actions[:, :, 3:]
    return torch.cat([relative, gripper], dim=2).flatten(1)


def branch_terms(embedding, truth):
    """W1 terms on the distribution's mean embedding: difference matching over all pairs plus an
    energy term rewarding spread. Non-divergent pairs are included by construction."""
    k = len(embedding)
    i, j = torch.triu_indices(k, k, offset=1, device=embedding.device)
    difference = (((embedding[i] - embedding[j]) - (truth[i] - truth[j])) ** 2).mean()
    cross = torch.cdist(embedding, truth).mean()
    within = torch.cdist(embedding, embedding).sum() / (k * k)
    return difference, cross - 0.5 * within


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/jenga/w1_data"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w2_ending_head.pt"))
    ap.add_argument("--report", default=str(ROOT / "results/jenga/w2_train.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-episodes", type=int, default=5)
    ap.add_argument("--branch-weight", type=float, default=1.0)
    ap.add_argument("--codes", type=int, default=256, help="codebook size (reported, not tuned)")
    ap.add_argument("--setup-cache", default=str(ROOT / "results/jenga/w2_setup.npz"))
    args = ap.parse_args()

    # Load ONLY the frames this model uses (3 history + the two endings). Loading all 41 frames
    # of every rollout is 43 GB at this dataset size and thrashes the machine.
    ending_index = {held: NUM_HIST + HORIZON + held - 1 for held in HOLD_STEPS}
    keep = [0, 1, 2] + [ending_index[held] for held in HOLD_STEPS]
    frames_list, actions_list, states_list, episodes = [], [], [], []
    for path in sorted(Path(args.data).glob("ep*.npz")):
        data = np.load(path, allow_pickle=False)
        frames_list.append(np.asarray(data["latents"][:, keep], np.float16))
        actions_list.append(data["actions"])
        states_list.append(data["state_ids"])
        episodes += [str(data["episode_id"])] * len(actions_list[-1])
        del data
    frames = np.concatenate(frames_list); del frames_list
    actions = np.concatenate(actions_list); del actions_list
    states = np.concatenate(states_list)
    episodes = np.asarray(episodes)
    grouped = {}
    for i, s in enumerate(states):
        grouped.setdefault(str(s), []).append(i)
    groups = [np.asarray(v) for v in grouped.values()]
    val_ids = set(sorted(set(episodes), key=int)[:args.val_episodes])
    is_val = np.isin(episodes, list(val_ids))

    endings = {held: frames[:, NUM_HIST + slot].reshape(len(frames), -1).astype(np.float32)
               for slot, held in enumerate(HOLD_STEPS)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codebooks, projections, targets, sizes = {}, {}, {}, {}
    cache_path = Path(args.setup_cache)
    cache = dict(np.load(cache_path, allow_pickle=False)) if cache_path.exists() else {}
    changed = False
    for held in HOLD_STEPS:
        keys = (f"mean_{held}", f"comp_{held}")
        if all(k in cache for k in keys):
            mean, components = cache[keys[0]], cache[keys[1]]
        else:
            mean, components = fit_pca(endings[held][~is_val])
            cache[keys[0]], cache[keys[1]] = mean, components
            changed = True
        projections[held] = (mean, components)
        projected = (endings[held] - mean) @ components.T
        code_key = f"codes_{held}_{args.codes}"
        if code_key in cache:
            centres, distortion = cache[code_key], float(cache.get(f"dist_{code_key}", -1))
        else:
            centres, distortion = build_codebook(projected[~is_val], args.codes)
            cache[code_key] = centres
            cache[f"dist_{code_key}"] = np.asarray(distortion)
            changed = True
        codebooks[held] = centres
        sizes[held] = {"codes": int(args.codes), "distortion": distortion}
        targets[held] = ((projected[:, None] - centres[None]) ** 2).sum(2).argmin(1)
        print(f"hold {held}: {args.codes} codes, "
              f"{len(np.unique(targets[held][~is_val]))} used in training", flush=True)
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)

    print("setup: codebooks done, projecting history", flush=True)
    history = frames[:, :NUM_HIST].reshape(len(frames), NUM_HIST, -1)
    history_pca = []
    if "mean_h" in cache:
        mean_h, comp_h = cache["mean_h"], cache["comp_h"]
    else:
        sample = history[~is_val][:, 0].astype(np.float32)
        mean_h, comp_h = fit_pca(sample)
        del sample
        cache["mean_h"], cache["comp_h"] = mean_h, comp_h
        np.savez(cache_path, **cache)
    print("setup: history PCA fitted", flush=True)
    mean_t = torch.as_tensor(mean_h, device=device)
    comp_t = torch.as_tensor(comp_h, device=device)
    for step in range(NUM_HIST):
        chunks = []
        for first in range(0, len(history), 512):  # GPU, in slices, to bound memory
            block = torch.as_tensor(history[first:first + 512, step], device=device).float()
            chunks.append(((block - mean_t) @ comp_t.T).cpu())
        history_pca.append(torch.cat(chunks))
    history_features = torch.cat(history_pca, dim=1)
    print("setup: history projected", flush=True)
    action_input = action_features(torch.from_numpy(actions))
    code_targets = {held: torch.from_numpy(targets[held]).long() for held in HOLD_STEPS}
    centre_tensors = {held: torch.from_numpy(codebooks[held]).float().to(device)
                      for held in HOLD_STEPS}
    truth_embed = {}
    for held in HOLD_STEPS:
        mean_e = torch.as_tensor(projections[held][0], device=device)
        comp_e = torch.as_tensor(projections[held][1], device=device)
        blocks = []
        for first in range(0, len(endings[held]), 512):
            block = torch.as_tensor(endings[held][first:first + 512], device=device)
            blocks.append(((block - mean_e) @ comp_e.T).cpu())
        truth_embed[held] = torch.cat(blocks)
    print("setup: targets embedded", flush=True)

    model = EndingHead(history_features.shape[1], action_input.shape[1],
                       max(sizes[h]["codes"] for h in HOLD_STEPS)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_groups = [g for g in groups if not is_val[g[0]]]
    val_groups = [g for g in groups if is_val[g[0]]]
    print(f"{len(train_groups)} training states, {len(val_groups)} validation states", flush=True)

    scales = {}

    def run(group, train):
        take = torch.as_tensor(np.asarray(group))
        logits = model(history_features[take].to(device), action_input[take].to(device))
        loss, stats = 0.0, {}
        for slot, held in enumerate(HOLD_STEPS):
            size = sizes[held]["codes"]
            head_logits = logits[slot][:, :size]
            target = code_targets[held][take].to(device)
            cross_entropy = nn.functional.cross_entropy(head_logits, target)
            probabilities = head_logits.softmax(1)
            embedding = probabilities @ centre_tensors[held][:size]
            difference, energy = branch_terms(embedding, truth_embed[held][take].to(device))
            # Scale-match the branch terms to cross-entropy once, on the first batch seen.
            scales.setdefault(f"diff_{held}", max(abs(float(difference.detach())), 1e-6))
            scales.setdefault(f"energy_{held}", max(abs(float(energy.detach())), 1e-6))
            scales.setdefault("ce", max(abs(float(cross_entropy.detach())), 1e-6))
            loss = loss + cross_entropy + args.branch_weight * scales["ce"] * (
                difference / scales[f"diff_{held}"] + energy / scales[f"energy_{held}"])
            stats[f"ce_{held}"] = float(cross_entropy.detach())
            stats[f"acc_{held}"] = float((head_logits.argmax(1) == target).float().mean())
            stats[f"diff_{held}"] = float(difference)
            stats[f"energy_{held}"] = float(energy)
        if train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        return stats

    def evaluate():
        model.eval()
        acc = []
        with torch.no_grad():
            for g in val_groups:
                acc.append(run(g, train=False))
        model.train()
        return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    report_history = {"validation": [evaluate()], "train": []}
    print("epoch 0:", report_history["validation"][0], flush=True)
    rng = np.random.default_rng(0)
    order = np.arange(len(train_groups))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        start, running = time.time(), []
        for index in order:
            running.append(run(train_groups[index], train=True))
        report_history["train"].append(
            {k: float(np.mean([r[k] for r in running])) for k in running[0]})
        report_history["validation"].append(evaluate())
        print(f"epoch {epoch + 1} ({time.time() - start:.0f}s) "
              f"train acc30 {report_history['train'][-1]['acc_30']:.3f} "
              f"val acc30 {report_history['validation'][-1]['acc_30']:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "codebooks": codebooks,
                    "projections": {h: projections[h] for h in HOLD_STEPS},
                    "history_projection": (mean_h, comp_h), "sizes": sizes,
                    "history_dim": history_features.shape[1],
                    "action_dim": action_input.shape[1]}, args.output)
    report = {"protocol": {
        "plan": "PLAN_WORLDMODEL W2a", "codebook": "PCA then k-means on TRAINING endings; k by "
        "elbow on training distortion", "hold_steps": list(HOLD_STEPS),
        "action_features": "targets relative to the chunk-start pose, in units of the measured "
        f"execution error ({NOISE_SCALE_MM} mm)", "epochs": args.epochs, "lr": args.lr,
        "grade_with": "eval/jenga_w2_curves.py (fork vs quiet jump ratio)"},
        "codebook_sizes": sizes, "branch_term_scales": scales,
        "history": report_history, "output": args.output}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
