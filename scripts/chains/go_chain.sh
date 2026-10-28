cd /home/corey/wksp/stable_lyap_filters
until [ "$(timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -n go,g3re | wc -l" 2>/dev/null || echo 1)" = "0" ]; do sleep 120; done
rsync -a -e "ssh -o BatchMode=yes" "fir:/scratch/ali313/slf/repo/results/program_b/bench2/opt3_m3int_*.part*.json" results/program_b/bench2/
ls results/program_b/bench2/opt3_m3int_go.part*.json | wc -l; ls results/program_b/bench2/opt3_m3int_g3re.part*.json | wc -l
python3 eval/go_bench.py merge go; python3 eval/go_bench.py merge g3re
while kill -0 1601095 2>/dev/null; do sleep 30; done
python3 eval/go_bench.py report go > results/jenga/go_offline.log 2>&1
python3 - > results/jenga/go_offline_g3re.log 2>&1 <<'PY'
import json, sys, numpy as np
sys.path[:0] = ["eval"]
import bench2 as b2
C, R = b2.load("m3int"); n = len(R)
T = {t: np.array(b2.truth_chunks("m3int", [(r["i"], np.asarray(r["chunk"], np.float32)) for r in json.load(open(f"results/program_b/bench2/opt3_m3int_{t}.json"))["rows"]])) for t in ("go", "g3re")}
ex = np.array(b2.truth_chunks("m3int", [(i, C["executed"][i]) for i in range(n)]))
rng = np.random.default_rng(1)
for col, lab in ((0, "neighbour"), (1, "red")):
    for a, b in (("go", "g3re"), ("g3re", "executed")):
        x = (T[a] if a != "executed" else ex)[:, col] - (T[b] if b != "executed" else ex)[:, col]
        bs = [x[rng.integers(n, size=n)].mean() for _ in range(10000)]
        print(lab, a, "-", b, "diff %.2f pp  ci90 [%.2f, %.2f]" % (100 * x.mean(), 100 * np.percentile(bs, 5), 100 * np.percentile(bs, 95)))
PY
echo done $(date)
