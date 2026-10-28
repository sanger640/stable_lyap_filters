cd /home/corey/wksp/stable_lyap_filters
until [ "$(timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -n g3oil | wc -l" 2>/dev/null || echo 1)" = "0" ]; do sleep 120; done
rsync -a -e "ssh -o BatchMode=yes" "fir:/scratch/ali313/slf/repo/results/program_b/bench2/opt3_m3int_g3oil.part*.json" results/program_b/bench2/
ls results/program_b/bench2/opt3_m3int_g3oil.part*.json | wc -l
python3 eval/go_bench.py merge g3oil
while kill -0 1603874 2>/dev/null; do sleep 60; done        # go_chain (g3re truth) uses the 30-process sim pool
python3 - <<'PY'
import json, sys, numpy as np
sys.path[:0] = ["eval"]
import bench2 as b2
C, R = b2.load("m3int")
G = {t: {r["i"]: r for r in json.load(open(f"results/program_b/bench2/opt3_m3int_{t}.json"))["rows"]} for t in ("g3re", "g3oil")}
I = sorted(set(G["g3re"]) & set(G["g3oil"]))
T = {t: np.array(b2.truth_chunks("m3int", [(i, np.asarray(G[t][i]["chunk"], np.float32)) for i in I])) for t in G}
pol = np.array(b2.truth_chunks("m3int", [(i, C["nominal"][i]) for i in I]))
out = {"decisions": len(I), "exploratory": True}
for t in G:
    pr = np.array([G[t][i]["progress"] for i in I])
    out[t] = {"neighbour": round(float(T[t][:, 0].mean()), 4), "red": round(float(T[t][:, 1].mean()), 4), "progress_median": float(np.median(pr)),
              "progress_mean": round(float(pr.mean()), 3), "hold_equivalent": round(float((pr < 0.1).mean()), 3), "seconds_median": float(np.median([G[t][i]["seconds"] for i in I]))}
out["policy"] = {"neighbour": round(float(pol[:, 0].mean()), 4), "red": round(float(pol[:, 1].mean()), 4)}
rng = np.random.default_rng(1)
for col, lab in ((0, "neighbour"), (1, "red")):
    x = T["g3oil"][:, col] - T["g3re"][:, col]; bs = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)]
    out[f"oil_minus_ens_{lab}_pp"] = [round(100 * x.mean(), 2), round(100 * np.percentile(bs, 5), 2), round(100 * np.percentile(bs, 95), 2)]
d = np.array([G["g3oil"][i]["progress"] - G["g3re"][i]["progress"] for i in I])
out["progress_oil_minus_ens"] = {"mean": round(float(d.mean()), 3), "better": int((d > 0.05).sum()), "worse": int((d < -0.05).sum())}
open("results/jenga/deep_dive/cert_ablation.json", "w").write(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
PY
echo done $(date)
