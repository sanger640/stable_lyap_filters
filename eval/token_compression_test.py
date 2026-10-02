"""Step 3 stage 0: which compressed DINOv2 latent keeps pose information and fork structure?

Frozen protocol: results/jenga/token_compression_protocol.json. Candidates: channel PCA (grid kept),
global PCA, a convolutional token autoencoder, DINOSAUR-style slots. Test 1 trains a pose readout on
each latent of the close-camera TRAIN tokens; test 2 runs the unchanged split rule on compressed
panel-2 ending tokens and compares with the state-position decisions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import jenga_spatial_readout as sr  # noqa: E402

PROTOCOL = ROOT / "results/jenga/token_compression_protocol.json"
OUT = ROOT / "results/program_b/compression"
RESULT = ROOT / "results/jenga/token_compression_result.json"
PANEL_DIR = ROOT / "results/program_b/visual_pipeline"
CANDIDATES = [("channel_pca", 8), ("channel_pca", 32), ("global_pca", 256), ("global_pca", 1024),
              ("autoencoder", 16), ("autoencoder", 64), ("slots", 8)]
SLOT_DIM, SLOT_ITERS, AE_WIDTH, FIT_EPOCHS = 256, 3, 256, 40
DEVICE = "cuda"


def name(kind, size):
    return f"{kind}_{size}"


# ----------------------------------------------------------------------------------- compressors
class ChannelPCA:
    """Per-patch projection of the channels onto their top-k principal directions."""

    def __init__(self, k):
        self.k = k

    def fit(self, grids):                                       # (N, H, W, C) float16, CPU
        c = grids.shape[-1]
        flat = grids.reshape(-1, c)
        total, outer, count = torch.zeros(c, dtype=torch.float64), torch.zeros(c, c, dtype=torch.float64), 0
        for first in range(0, len(flat), 1 << 18):
            x = torch.as_tensor(flat[first:first + (1 << 18)], device=DEVICE).double()
            total += x.sum(0).cpu(); outer += (x.T @ x).cpu(); count += len(x)
        self.mean = total / count
        cov = outer / count - torch.outer(self.mean, self.mean)
        vals, vecs = torch.linalg.eigh(cov)
        self.basis = vecs[:, torch.argsort(vals, descending=True)[:self.k]].float()   # (C, k)
        self.mean = self.mean.float()
        return self

    def transform(self, grids):
        if not hasattr(self, "_gpu"):
            self._gpu = (self.mean.to(DEVICE), self.basis.to(DEVICE))
        mean, basis = self._gpu
        out = []
        for first in range(0, len(grids), 512):
            x = torch.as_tensor(grids[first:first + 512], device=DEVICE).float()
            out.append(((x - mean) @ basis).cpu())
        return torch.cat(out).numpy()                           # (N, H, W, k)


class GlobalPCA:
    """Exact PCA of whole flattened frames through the n x n Gram matrix (feature dim chunked)."""

    def __init__(self, k):
        self.k = k

    def _chunks(self, d):
        step = 1 << 15
        return [slice(first, min(first + step, d)) for first in range(0, d, step)]

    def fit(self, grids):
        x = grids.reshape(len(grids), -1)
        n, d = x.shape
        self.mean = torch.zeros(d)
        gram = torch.zeros(n, n, device=DEVICE)
        for ch in self._chunks(d):
            block = torch.as_tensor(x[:, ch], device=DEVICE).float()
            mu = block.mean(0)
            self.mean[ch] = mu.cpu()
            block -= mu
            gram += block @ block.T
        gram = gram.cpu().double()                              # eigh on CPU: the n x n float64 copy
        torch.cuda.empty_cache()                                # does not fit beside other GPU jobs
        vals, vecs = torch.linalg.eigh(gram)
        del gram
        top = torch.argsort(vals, descending=True)[:self.k]
        u, s = vecs[:, top].float().to(DEVICE), vals[top].clamp_min(1e-12).sqrt().float().to(DEVICE)
        self.components = torch.zeros(self.k, d)
        for ch in self._chunks(d):
            block = torch.as_tensor(x[:, ch], device=DEVICE).float() - self.mean[ch].to(DEVICE)
            self.components[:, ch] = ((u.T @ block) / s[:, None]).cpu()
        self.explained = float(vals[top].sum() / vals.clamp_min(0).sum())
        return self

    def transform(self, grids):
        if not hasattr(self, "_gpu"):
            self._gpu = (self.mean.to(DEVICE), self.components.to(DEVICE))
        mean, comps = self._gpu
        x = grids.reshape(len(grids), -1)
        z = torch.zeros(len(x), self.k, device=DEVICE)
        for first in range(0, len(x), 256):
            block = torch.as_tensor(x[first:first + 256], device=DEVICE).float() - mean
            z[first:first + 256] = block @ comps.T
        return z.cpu().numpy()                                  # (N, k)


class TokenStats:
    def __init__(self, grids):
        flat = grids.reshape(-1, grids.shape[-1])
        pick = np.random.default_rng(0).choice(len(flat), min(len(flat), 1 << 20), replace=False)
        sample = torch.as_tensor(flat[np.sort(pick)]).float()
        self.mean, self.std = sample.mean(0), sample.std(0).clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class ConvAE(nn.Module):
    def __init__(self, side, channels, code):
        super().__init__()
        downs = int(round(math.log2(side / 7)))
        enc = [nn.Conv2d(channels, AE_WIDTH, 1), nn.GELU()]
        dec = [nn.Conv2d(code, AE_WIDTH, 1), nn.GELU()]
        for _ in range(downs):
            enc += [nn.Conv2d(AE_WIDTH, AE_WIDTH, 3, stride=2, padding=1), nn.GELU()]
            dec += [nn.ConvTranspose2d(AE_WIDTH, AE_WIDTH, 4, stride=2, padding=1), nn.GELU()]
        self.encoder = nn.Sequential(*enc, nn.Conv2d(AE_WIDTH, code, 1))
        self.decoder = nn.Sequential(*dec, nn.Conv2d(AE_WIDTH, channels, 1))

    def encode(self, grid):                                     # (B, H, W, C) -> (B, 7, 7, code)
        return self.encoder(grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def forward(self, grid):
        return self.decoder(self.encode(grid).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SlotAE(nn.Module):
    """DINOSAUR-style: slot attention over frozen features, spatial-broadcast MLP decoder."""

    def __init__(self, tokens, channels, slots, dim=SLOT_DIM, hidden=512):
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, dim), nn.GELU(),
                                   nn.Linear(dim, dim))
        self.init = nn.Parameter(torch.randn(slots, dim) * dim ** -0.5)
        self.norm_in, self.norm_slot, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.pos = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.decoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, channels + 1))
        self.dim = dim

    def encode(self, tokens):                                   # (B, N, C) -> (B, K, D)
        x = self.norm_in(self.embed(tokens))
        k, v = self.k(x), self.v(x)
        slots = self.init.expand(len(x), -1, -1)
        for _ in range(SLOT_ITERS):
            q = self.q(self.norm_slot(slots))
            attn = torch.softmax(k @ q.transpose(1, 2) / self.dim ** 0.5, dim=-1) + 1e-8   # (B, N, K)
            attn = attn / attn.sum(1, keepdim=True)
            updates = attn.transpose(1, 2) @ v
            slots = self.gru(updates.reshape(-1, self.dim), slots.reshape(-1, self.dim)).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

    def forward(self, tokens):
        slots = self.encode(tokens)
        out = self.decoder(slots[:, :, None, :] + self.pos[None, None])    # (B, K, N, C+1)
        alpha = torch.softmax(out[..., -1], dim=1)
        return (alpha[..., None] * out[..., :-1]).sum(1)


class NeuralCompressor:
    def __init__(self, kind, size):
        self.kind, self.size = kind, size

    def _input(self, grids, first, count):
        x = self.stats(torch.as_tensor(grids[first:first + count], device=DEVICE).float())
        return x if self.kind == "autoencoder" else x.flatten(1, 2)

    def fit(self, grids, seed=0):
        torch.manual_seed(seed)
        n, side, _, c = grids.shape
        self.stats = TokenStats(grids)
        self.net = (ConvAE(side, c, self.size) if self.kind == "autoencoder"
                    else SlotAE(side * side, c, self.size)).to(DEVICE)
        batch = 64 if self.kind == "autoencoder" or side <= 14 else 16
        lr = 1e-3 if self.kind == "autoencoder" else 4e-4
        steps = FIT_EPOCHS * math.ceil(n / batch)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for epoch in range(FIT_EPOCHS):
            total = 0.0
            for idx in torch.randperm(n).split(batch):
                idx = np.sort(idx.numpy())
                x = self.stats(torch.as_tensor(grids[idx], device=DEVICE).float())
                x = x if self.kind == "autoencoder" else x.flatten(1, 2)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ((self.net(x).float() - x) ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                total += loss.item() * len(idx)
            if (epoch + 1) % 10 == 0:
                print(f"    {self.kind}_{self.size} epoch {epoch + 1}: recon {total / n:.4f}", flush=True)
        self.recon = total / n
        self.net.eval()
        return self

    def transform(self, grids):
        out = []
        with torch.no_grad():
            for first in range(0, len(grids), 128):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(self.net.encode(self._input(grids, first, 128)).float().cpu())
        return torch.cat(out).numpy()


def make(kind, size):
    return {"channel_pca": ChannelPCA, "global_pca": GlobalPCA}.get(kind, lambda s: NeuralCompressor(kind, s))(size)


def numbers_per_frame(kind, size, side):
    return {"channel_pca": side * side * size, "global_pca": size, "autoencoder": 49 * size,
            "slots": size * SLOT_DIM}[kind]


# ------------------------------------------------------------------------------- test 1: pose
class VectorReadout(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                 nn.Linear(1024, sr.POSE))

    def forward(self, z):
        return self.net(z)


class SetReadout(nn.Module):
    def __init__(self, dim, width=256):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.summary = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(width, sr.POSE)

    def forward(self, slots):
        x = torch.cat([self.summary.expand(len(slots), -1, -1), self.inp(slots)], 1)
        return self.out(self.body(x)[:, 0])


def standardise(train, held):
    axes = tuple(range(train.ndim - 1))
    mean, std = train.mean(axes, keepdims=True), train.std(axes, keepdims=True) + 1e-6
    return ((train - mean) / std).astype(np.float32), ((held - mean) / std).astype(np.float32)


def pose_readout(kind, z_train, y_train, z_held, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if kind in ("full", "channel_pca", "autoencoder"):
        model = sr.KeypointHead(z_train.shape[-1])
    elif kind == "global_pca":
        model = VectorReadout(z_train.shape[-1])
    else:
        model = SetReadout(z_train.shape[-1])
    model = model.to(DEVICE)
    x_train, x_held = torch.from_numpy(z_train), torch.from_numpy(z_held)
    steps = sr.EPOCHS * math.ceil(len(x_train) / sr.BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(sr.EPOCHS):
        order = torch.randperm(len(x_train))
        for first in range(0, len(order), sr.BATCH):
            idx = order[first:first + sr.BATCH]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x_train[idx].to(DEVICE).float())
            loss = ((pred.float() - y_train[idx].to(DEVICE)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x_held[i:i + 256].to(DEVICE).float()).float().cpu()
                          for i in range(0, len(x_held), 256)])


def test1(candidates):
    done = OUT / "test1.json"
    rows = json.loads(done.read_text()) if done.exists() else {}
    grids, states, ids = sr.load(sr.ENCODED / "enc_dinov2_392_close")
    unique = sorted(set(ids)); val = set(unique[:max(1, int(len(unique) * 0.12))])
    held = np.array([i in val for i in ids])
    y = torch.from_numpy(states[:, :sr.POSE])
    mean, std = y[~held].mean(0), y[~held].std(0).clamp_min(1e-6)
    y_train = (y[~held] - mean) / std
    train_grids, held_grids = grids[~held], grids[held]
    import jenga_v1_probe as vp
    for kind, size in [("full", 0)] + candidates:
        key = "full" if kind == "full" else name(kind, size)
        if key in rows:
            continue
        print(f"test1 {key}", flush=True)
        extra = {}
        if kind == "full":
            z_train, z_held = train_grids, held_grids
        else:
            comp = make(kind, size).fit(train_grids)
            z_train, z_held = standardise(comp.transform(train_grids), comp.transform(held_grids))
            extra = {k: getattr(comp, k) for k in ("explained", "recon") if hasattr(comp, k)}
        pose = pose_readout(kind, z_train, y_train, z_held) * std + mean
        full = torch.from_numpy(states[held]).clone()
        predicted = full.clone(); predicted[:, :sr.POSE] = pose
        g = vp.geometry_errors(predicted, full)
        rows[key] = {"relative_position_mm": g["relative_position_mm"], "orientation_deg": g["orientation_deg"],
                     "absolute_position_mm": g["absolute_position_mm"],
                     "numbers_per_frame_28x28": 28 * 28 * 384 if kind == "full" else numbers_per_frame(kind, size, 28),
                     **extra}
        print(f"  {key}: {rows[key]}", flush=True)
        done.write_text(json.dumps(rows, indent=1) + "\n")
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------ test 2: forks
def panel_keys():
    from visual_pipeline_panel2 import panel2_selection
    panel, _ = panel2_selection()
    return sorted(panel, key=lambda k: (int(k[0]), k[1]))


def encode_panel(keys):
    """DINOv2 196 tokens of every ending frame, exactly as Step 2, plus the full-token decisions."""
    from torchvision.transforms import functional as TF
    from counterfactual_monitor import persistent_mode_alarm
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    tok_dir = OUT / "panel2_tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    encoder = None
    for number, (e, i) in enumerate(keys, 1):
        path = tok_dir / f"ep{e}_s{i}.npz"
        if path.exists():
            continue
        if encoder is None:
            encoder = load_world_model(DEFAULT_CHECKPOINT, DEVICE).encoder
        saved = {}
        with np.load(PANEL_DIR / f"ep{e}_s{i}_frames.npz") as f, np.load(PANEL_DIR / f"ep{e}_s{i}.npz") as s:
            for cond in ("exact", "aware"):
                frames, toks = f[cond], []
                for t in range(2):
                    x = torch.as_tensor(frames[:, t], device=DEVICE).permute(0, 3, 1, 2).float() / 255
                    x = (TF.resize(x, [196, 196], antialias=True) - 0.5) / 0.5
                    with torch.inference_mode():
                        toks.append(encoder.forward(x).float().cpu().numpy())
                saved[f"{cond}_tokens"] = np.stack(toks, 1).astype(np.float16)          # (64, 2, 196, 384)
                flat = [tk.reshape(len(tk), -1) for tk in toks]
                saved[f"{cond}_full"] = np.array(persistent_mode_alarm(flat[0], flat[1]).alarm)
                saved[f"{cond}_state"] = np.array(decide(*trace_endings(s[cond], NON_ROBOT["jenga"]))[0])
        np.savez(path.with_suffix(".tmp.npz"), **saved)
        path.with_suffix(".tmp.npz").replace(path)
        print(f"encoded {number}/{len(keys)}", flush=True)
    return tok_dir


def align_slots(z):
    """(64, K, D) per hold: permute each rollout's slots to best match rollout 0 (cosine)."""
    from scipy.optimize import linear_sum_assignment
    ref = z[0] / (np.linalg.norm(z[0], axis=-1, keepdims=True) + 1e-9)
    out = np.empty_like(z)
    for j, slots in enumerate(z):
        unit = slots / (np.linalg.norm(slots, axis=-1, keepdims=True) + 1e-9)
        _, col = linear_sum_assignment(-(ref @ unit.T))
        out[j] = slots[col]
    return out


def compressed_decision(kind, comp, tokens):
    """tokens (64, 2, 196, 384) float16 -> alarm of the split rule on compressed endings."""
    from counterfactual_monitor import persistent_mode_alarm
    flat = []
    for t in range(2):
        z = comp.transform(tokens[:, t].reshape(64, 14, 14, -1))
        if kind == "slots":
            z = align_slots(z)
        flat.append(z.reshape(64, -1).astype(np.float64))
    return bool(persistent_mode_alarm(flat[0], flat[1]).alarm)


def test2(candidates, keys, tok_dir):
    import fragility_grading_dev as fg
    done = OUT / "test2.json"
    rows = json.loads(done.read_text()) if done.exists() else {}
    episodes = sorted({e for e, _ in keys}, key=int)
    fold_of = {e: n % 2 for n, e in enumerate(episodes)}
    ref = {c: {k: None for k in ("state", "full")} for c in ("exact", "aware")}
    for c in ref:
        for k in ref[c]:
            ref[c][k] = []
    for e, i in keys:
        with np.load(tok_dir / f"ep{e}_s{i}.npz") as f:
            for c in ref:
                ref[c]["state"].append(bool(f[f"{c}_state"])); ref[c]["full"].append(bool(f[f"{c}_full"]))
    rows["full"] = {c: {"kappa_vs_state": fg.kappa(ref[c]["full"], ref[c]["state"]),
                        "alarms": int(np.sum(ref[c]["full"])), "state_alarms": int(np.sum(ref[c]["state"]))}
                    for c in ref}
    rows["full"]["numbers_per_frame_14x14"] = 196 * 384
    for kind, size in candidates:
        key = name(kind, size)
        if key in rows:
            continue
        print(f"test2 {key}", flush=True)
        decisions = {c: [None] * len(keys) for c in ("exact", "aware")}
        extra = {}
        for fold in (0, 1):
            fit_keys = [k for k in keys if fold_of[k[0]] != fold]
            fit = np.concatenate([np.load(tok_dir / f"ep{e}_s{i}.npz")[f"{c}_tokens"].reshape(-1, 14, 14, 384)
                                  for e, i in fit_keys for c in ("exact", "aware")])
            comp = make(kind, size).fit(fit)
            del fit
            extra[f"fold{fold}"] = {k: getattr(comp, k) for k in ("explained", "recon") if hasattr(comp, k)}
            for n, (e, i) in enumerate(keys):
                if fold_of[e] != fold:
                    continue
                with np.load(tok_dir / f"ep{e}_s{i}.npz") as f:
                    for c in decisions:
                        decisions[c][n] = compressed_decision(kind, comp, f[f"{c}_tokens"])
            torch.cuda.empty_cache()
        rows[key] = {c: {"kappa_vs_state": fg.kappa(decisions[c], ref[c]["state"]),
                         "agreement_with_full": float(np.mean(np.array(decisions[c]) == np.array(ref[c]["full"]))),
                         "alarms": int(np.sum(decisions[c]))} for c in decisions}
        rows[key]["numbers_per_frame_14x14"] = numbers_per_frame(kind, size, 14)
        rows[key].update(extra)
        print(f"  {key}: {rows[key]}", flush=True)
        done.write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def select(t1, t2, candidates):
    base1 = t1["full"]["relative_position_mm"]
    order = ["channel_pca", "global_pca", "autoencoder", "slots"]
    verdict = {}
    for kind, size in candidates:
        key = name(kind, size)
        p1 = t1[key]["relative_position_mm"] <= 1.25 * base1
        p2 = all(t2[key][c]["kappa_vs_state"] >= t2["full"][c]["kappa_vs_state"] - 0.10 for c in ("exact", "aware"))
        verdict[key] = {"test1_pass": bool(p1), "test2_pass": bool(p2), "qualifies": bool(p1 and p2),
                        "numbers_per_frame_14x14": numbers_per_frame(kind, size, 14)}
    qualified = [(verdict[name(k, s)]["numbers_per_frame_14x14"], order.index(k), name(k, s))
                 for k, s in candidates if verdict[name(k, s)]["qualifies"]]
    return verdict, (min(qualified)[2] if qualified else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("test1", "test2"), default=None)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    keys = panel_keys()
    t2 = t1 = None
    if args.only != "test1":
        t2 = test2(CANDIDATES, keys, encode_panel(keys))
    if args.only != "test2":
        t1 = test1(CANDIDATES)
    if t1 is None or t2 is None:
        return
    verdict, chosen = select(t1, t2, CANDIDATES)
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "implementation_notes": "compressed latents are standardised per channel with training statistics before the pose readout; the full-token arm uses raw tokens exactly as the frozen perception model",
        "panel_states": len(keys),
        "test1_pose": t1, "test2_forks": t2, "verdict": verdict, "chosen_latent": chosen,
        "sanity_full_reproduces_step2": bool(abs(t2["full"]["exact"]["kappa_vs_state"] - 0.681) < 0.005
                                            and abs(t2["full"]["aware"]["kappa_vs_state"] - 0.903) < 0.005),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("verdict", "chosen_latent", "sanity_full_reproduces_step2")}, indent=1))


if __name__ == "__main__":
    main()
