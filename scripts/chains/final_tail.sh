cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/cbf_final.log
for p in 2018433 2018434 2018435 2018436; do
  while kill -0 $p 2>/dev/null; do sleep 60; done
done
echo "local workers done $(date)" >> $L
scp -q gpu5090:wksp/stable_lyap_filters/results/jenga/paired_cbf_test2x.jsonl /tmp/remote_cbf_test2x.jsonl
python3 - <<'EOF'
import json
loc = {json.loads(l)["episode"]: l for l in open("results/jenga/paired_cbf_test2x.jsonl")}
for l in open("/tmp/remote_cbf_test2x.jsonl"):
    e = json.loads(l)["episode"]
    if e not in loc:
        loc[e] = l
open("results/jenga/paired_cbf_test2x.jsonl", "w").write("".join(loc[e] if loc[e].endswith("\n") else loc[e] + "\n" for e in sorted(loc)))
print("merged", len(loc))
EOF
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/policy_rollouts/paired_cbf_test2x/ results/program_b/policy_rollouts/paired_cbf_test2x/ >> $L 2>&1
python3 eval/cbf_final_test.py analyse >> $L 2>&1
echo "final done $(date)" >> $L
