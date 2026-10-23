"""Full-token DINO-WM world model for Jenga (docs/FILTER_DEEP_DIVE.md step 2).

Architecture = DINO-WM (Zhou et al. 2025; code vendored in src/models/dinowm, unchanged): frozen DINOv2 ViT-S/14 patch
tokens (196 px -> 14x14x384), each patch token concatenated with a proprio embedding (10) and an action embedding (10)
(concat_dim = 1, dim 404), a 6-layer ViT predictor over a 3-frame history with the frame-causal mask, predicting the next
frame's tokens. Initialised from the project's DINO-WM checkpoint (data/jenga/world_model.pt): predictor and proprio
embedding as is; the action embedding is widened from 1 to 2 actions per model step (two copies of the pretrained weights,
each halved, so it starts as the mean of the two actions). Deterministic, as DINO-WM.
Training: one-step prediction loss on visual + proprio dims (as DINO-WM), with self-rollout inputs on half the batches
(the recipe that worked for our own models), on the full-token counterfactual data from eval/tok_data.py.
usage: python3 eval/tokwm.py train <data_dir> <out.pt> [steps] ; val <data_dir> <model.pt>
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "src/models")]
H, P, D, E = 3, 196, 384, 10                     # history frames, patches, token dim, proprio/action embedding dim


def _torch():
    import torch
    return torch


class TokWM:
    def __init__(self, device="cuda"):
        torch = _torch(); nn = torch.nn
        from dinowm import install_import_shim
        install_import_shim()
        import dinowm.vit as vit
        vit.NUM_FRAMES, vit.NUM_PATCHES = H, P
        ck = torch.load(ROOT / "data/jenga/world_model.pt", map_location="cpu", weights_only=False)
        self.pred = ck["predictor"].to(device)
        self.prop_enc = ck["proprio_encoder"].to(device)
        a1 = ck["action_encoder"]
        self.act_enc = nn.Conv1d(8, E, 1).to(device)
        with torch.no_grad():
            w = a1.patch_embed.weight.data                                          # (10, 4, 1)
            self.act_enc.weight.copy_(torch.cat([w, w], 1) / 2); self.act_enc.bias.copy_(a1.patch_embed.bias.data)
        self.device = device
        self.stats = None                                                           # set by fit_stats / load

    def params(self):
        return list(self.pred.parameters()) + list(self.prop_enc.parameters()) + list(self.act_enc.parameters())

    def train(self, on=True):
        for m in (self.pred, self.prop_enc, self.act_enc):
            m.train(on)

    # ---- normalisation
    def fit_stats(self, props, acts):
        p = props.reshape(-1, 4); a = acts.reshape(-1, 4)
        self.stats = {"p_mean": p.mean(0), "p_std": p.std(0) + 1e-6, "a_mean": a.mean(0), "a_std": a.std(0) + 1e-6}

    def _np(self, k):
        return _torch().as_tensor(np.asarray(self.stats[k], np.float32), device=self.device)

    def enc_prop(self, prop):                               # (B, T, 4) -> (B, T, E)
        x = (prop - self._np("p_mean")) / self._np("p_std")
        return self.prop_enc.patch_embed(x.transpose(1, 2)).transpose(1, 2)

    def enc_act(self, act):                                 # (B, T, 8) -> (B, T, E)
        x = (act.reshape(*act.shape[:2], 2, 4) - self._np("a_mean")) / self._np("a_std")
        return self.act_enc(x.reshape(*act.shape[:2], 8).transpose(1, 2)).transpose(1, 2)

    def tokens(self, vis, pe, ae):                          # vis (B, T, P, D), pe/ae (B, T, E) -> (B, T, P, D + 2E)
        T = vis.shape[1]
        return _torch().cat([vis, pe[:, :, None].expand(-1, -1, P, -1), ae[:, :, None].expand(-1, -1, P, -1)], -1)

    def predict(self, z):                                   # z (B, H, P, D + 2E) -> next frame (vis (B, P, D), pe (B, E))
        B = z.shape[0]
        out = self.pred(z.reshape(B, H * P, D + 2 * E)).reshape(B, H, P, D + 2 * E)[:, -1]
        return out[..., :D], out[..., D:D + E].mean(1)

    def rollout(self, vis0, prop0, acts, keep=(8, 18), grad=False):
        """vis0 (B, P, D) start tokens, prop0 (B, 4), acts (B, S, 8) -> {step: vis (B, P, D)} after step t (0-based) in keep.
        History before the start is padded with the start frame (as in training)."""
        torch = _torch()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            pe0 = self.enc_prop(prop0[:, None])[:, 0]; ae = self.enc_act(acts)
            vis = [vis0] * H; pes = [pe0] * H; aes = [ae[:, 0]] * H; out = {}
            for t in range(acts.shape[1]):
                aes = aes[1:] + [ae[:, t]]
                z = self.tokens(torch.stack(vis[-H:], 1), torch.stack(pes[-H:], 1), torch.stack(aes, 1))
                v, p_ = self.predict(z)
                vis.append(v); pes.append(p_)
                if t in keep:
                    out[t] = v
            return out

    def save(self, path):
        torch = _torch()
        torch.save({"pred": self.pred.state_dict(), "prop_enc": self.prop_enc.state_dict(), "act_enc": self.act_enc.state_dict(),
                    "stats": {k: np.asarray(v).tolist() for k, v in self.stats.items()}}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        torch = _torch(); m = cls(device); s = torch.load(path, map_location=device, weights_only=False)
        m.pred.load_state_dict(s["pred"]); m.prop_enc.load_state_dict(s["prop_enc"]); m.act_enc.load_state_dict(s["act_enc"])
        m.stats = {k: np.asarray(v, np.float32) for k, v in s["stats"].items()}; m.train(False)
        for q in m.params():
            q.requires_grad_(False)
        return m


def deq(q, s):
    return q.astype(np.float32) * s.astype(np.float32)[..., None]


class Data:
    """Loads tok_data files: per rollout k, frames 0..19 (0 = perturbed start), props 0..19, actions 38 -> 19 x 8."""

    def __init__(self, d, max_files=None):
        fs = sorted(Path(d).glob("ep*.npz"))[:max_files]
        V, PR, A = [], [], []
        for f in fs:
            with np.load(f) as z:
                v = np.concatenate([z["q_start"][:, None], z["q"]], 1); s = np.concatenate([z["s_start"][:, None], z["s"]], 1)
                V.append((v, s)); PR.append(z["props"]); A.append(z["actions"].reshape(len(v), 19, 8))
        self.q = np.concatenate([v for v, _ in V]); self.s = np.concatenate([s for _, s in V])
        self.props = np.concatenate(PR).astype(np.float32); self.acts = np.concatenate(A).astype(np.float32)
        print("data", d, "rollouts", len(self.q), flush=True)

    def batch(self, rng, n, device):
        torch = _torch()
        k = rng.integers(len(self.q), size=n); t = rng.integers(19, size=n)             # predict frame t + 1 from <= t
        idx = np.clip(t[:, None] + np.arange(-H + 1, 1)[None], 0, None)                # (n, H) history frames, padded with 0
        vis = deq(self.q[k[:, None], idx], self.s[k[:, None], idx])
        tgt = deq(self.q[k, t + 1], self.s[k, t + 1])
        acts = self.acts[k[:, None], idx]                     # action j leads from frame j to j + 1; padding repeats step 0
        f = lambda x: torch.as_tensor(x, device=device)
        return f(vis), f(self.props[k[:, None], idx]), f(acts), f(tgt), f(self.props[k, t + 1]), (k, t)


def train(data_dir, out, steps=30000, batch=48, lr=1e-4):
    torch = _torch(); rng = np.random.default_rng(0); torch.manual_seed(0)
    tr = Data(Path(data_dir) / "train"); va = Data(Path(data_dir) / "val")
    m = TokWM(); m.fit_stats(tr.props, tr.acts); m.train(True)
    opt = torch.optim.AdamW(m.params(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    log = []
    for step in range(steps):
        vis, pr, ac, tgt, ptgt, (k, t) = tr.batch(rng, batch, "cuda")
        if step % 2 == 1:                                    # self-rollout input: replace the newest history frame by a prediction
            with torch.no_grad():
                m.train(False)
                ok = t >= 1
                if ok.any():
                    kk, tt = k[ok], t[ok] - 1
                    idx = np.clip(tt[:, None] + np.arange(-H + 1, 1)[None], 0, None)
                    pv = torch.as_tensor(deq(tr.q[kk[:, None], idx], tr.s[kk[:, None], idx]), device="cuda")
                    pa = torch.as_tensor(tr.acts[kk[:, None], idx], device="cuda")
                    pp = torch.as_tensor(tr.props[kk[:, None], idx], device="cuda")
                    v_, _ = m.predict(m.tokens(pv, m.enc_prop(pp), m.enc_act(pa)))
                    vis[torch.as_tensor(ok, device="cuda"), -1] = v_
                m.train(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v, pe = m.predict(m.tokens(vis, m.enc_prop(pr), m.enc_act(ac)))
            loss_v = (v.float() - tgt).pow(2).mean(); loss_p = (pe.float() - m.enc_prop(ptgt[:, None])[:, 0].detach()).pow(2).mean()
            loss = loss_v + loss_p
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.params(), 1.0); opt.step(); sched.step()
        if step % 1000 == 999 or step == steps - 1:
            vl = val_loss(m, va); m.train(True)
            log.append({"step": step + 1, "loss_v": float(loss_v), "loss_p": float(loss_p), **vl}); print(log[-1], flush=True)
            m.save(out)
    Path(str(out) + ".log.json").write_text(json.dumps(log, indent=0))


def val_loss(m, va, n=512):
    torch = _torch(); rng = np.random.default_rng(1); m.train(False); tot, base = [], []
    with torch.no_grad():
        for _ in range(n // 64):
            vis, pr, ac, tgt, ptgt, _ = va.batch(rng, 64, "cuda")
            v, _ = m.predict(m.tokens(vis, m.enc_prop(pr), m.enc_act(ac)))
            tot.append(float((v - tgt).pow(2).mean())); base.append(float((vis[:, -1] - tgt).pow(2).mean()))
    return {"val_mse": float(np.mean(tot)), "val_copy_last_mse": float(np.mean(base))}


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "train":
        train(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:5]))
    elif st == "val":
        m = TokWM.load(sys.argv[3]); print(val_loss(m, Data(Path(sys.argv[2]) / "val")))
