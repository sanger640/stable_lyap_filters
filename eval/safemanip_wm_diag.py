"""SafeManip 2b-WM diagnosis after the gate failed (docs/BENCHMARK_DD.md: "If neither passes: diagnose as for Physion").
Per view, on the held-out eval points, imagining with the REAL replays' own executed actions (exact + 32 noisy):
  - ending error: imagined end-of-hold latent vs the real one, against the copy baseline (start latent as the ending)
  - spread: per-point std of endings across the 32 noisy replays, imagined vs real
  - split test on REAL endings (sanity: should reproduce check 2c) and on imagined endings, AUROC vs real fork
  - bad-vs-good readout: distance of each replay's ending from the exact replay's, imagined vs real, AUROC vs bad replay
And on training points: rollout error at the last model step vs copy (fit check).
usage: SM_VIEW=monitor python3 eval/safemanip_wm_diag.py
"""
import json

import numpy as np
import torch

import safemanip_wm as S


def main():
    import cem_filter as cf
    import step3_world_model as wm
    from wm_libero import auroc
    S._patch_wm()
    model = wm.WorldModel.load(S.OUT / "model_SM.pt")
    rows = {}
    for jf in (S.DATA / "eval").glob("s*/*.json"):
        for r in json.loads(jf.read_text())["rows"]:
            rows[f"{jf.parent.name}_{r['task']}_s{jf.parent.name[1:]}_e{r['episode']}_t{r['t']}.npz"] = r

    def roll(z0, inp):
        z = z0[None].expand(len(inp), -1, -1, -1).contiguous()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(S.STEPS):
                z = model.step(z, inp[:, t]).float()
        return z

    res = []
    for f in sorted((S.ENC / "eval").glob("*.npz")):
        r = rows.get(f.name)
        if r is None:
            continue
        d = np.load(f); src = np.load(S.DATA / "eval" / f.name.split("_", 1)[0] / f.name.split("_", 1)[1])
        zr = torch.as_tensor(d["z"], device="cuda").float()                    # (33, 2, 28, 28, 8): start, end of hold
        inp = torch.as_tensor(d["inputs"], device="cuda")                       # (33, 15, 41) from the real actions
        zi = roll(zr[0, 0], inp)                                                # (33, 28, 28, 8)
        real_end, start = zr[:, 1], zr[0, 0]
        err = float(((zi - real_end) ** 2).mean()); copy = float(((start[None] - real_end) ** 2).mean())
        ok_r, _, _ = cf.split_test_gpu(real_end[1:].reshape(1, 32, -1)); ok_i, _, _ = cf.split_test_gpu(zi[1:].reshape(1, 32, -1))
        dr = ((real_end[1:] - real_end[0]) ** 2).flatten(1).mean(1).cpu().numpy()
        di = ((zi[1:] - zi[0]) ** 2).flatten(1).mean(1).cpu().numpy()
        res.append({"file": f.name, "fork": 2 <= r["n_bad"] <= 30, "n_bad": r["n_bad"], "err": err, "copy": copy,
                    "spread_real": float(real_end[1:].std(0).mean()), "spread_imag": float(zi[1:].std(0).mean()),
                    "split_real": bool(ok_r[0]), "split_imag": bool(ok_i[0]),
                    "bad": r["bad_replays"][1:], "dist_real": dr.tolist(), "dist_imag": di.tolist()})
    fork = np.array([x["fork"] for x in res])
    bad = np.concatenate([x["bad"] for x in res]).astype(bool)
    out = {"view": S.VIEW, "points": len(res), "real_forks": int(fork.sum()),
           "ending_mse_model": float(np.mean([x["err"] for x in res])), "ending_mse_copy": float(np.mean([x["copy"] for x in res])),
           "spread_real_mean": float(np.mean([x["spread_real"] for x in res])), "spread_imag_mean": float(np.mean([x["spread_imag"] for x in res])),
           "spread_corr": float(np.corrcoef([x["spread_real"] for x in res], [x["spread_imag"] for x in res])[0, 1]),
           "AUROC_spread_real_vs_fork": auroc(np.array([x["spread_real"] for x in res]), fork),
           "AUROC_spread_imag_vs_fork": auroc(np.array([x["spread_imag"] for x in res]), fork),
           "split_real_rate_forks_nonforks": [float(np.mean([x["split_real"] for x in res if x["fork"]])), float(np.mean([x["split_real"] for x in res if not x["fork"]]))],
           "split_imag_rate_forks_nonforks": [float(np.mean([x["split_imag"] for x in res if x["fork"]])), float(np.mean([x["split_imag"] for x in res if not x["fork"]]))],
           "bad_replays": int(bad.sum()),
           "AUROC_dist_real_vs_bad_replay": auroc(np.concatenate([x["dist_real"] for x in res]), bad),
           "AUROC_dist_imag_vs_bad_replay": auroc(np.concatenate([x["dist_imag"] for x in res]), bad)}
    tr = []                                                                     # fit check on training points
    for f in sorted((S.ENC / "train").glob("*.npz"))[::20]:
        d = np.load(f); z = torch.as_tensor(d["z"], device="cuda").float(); inp = torch.as_tensor(d["inputs"], device="cuda")
        zi = roll(z[0, 0], inp)
        tr.append((float(((zi - z[:, -1]) ** 2).mean()), float(((z[0, 0][None] - z[:, -1]) ** 2).mean())))
    out["train_last_mse_model_copy"] = [float(np.mean([a for a, _ in tr])), float(np.mean([b for _, b in tr])), len(tr)]
    (S.OUT / "diag_rows.json").write_text(json.dumps(res)); (S.OUT / "diag_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
