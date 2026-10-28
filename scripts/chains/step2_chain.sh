cd /home/corey/wksp/stable_lyap_filters
T=paired_record_dev2x
while pgrep -f "eval/policy_rollout.py paired record" > /dev/null || ssh gpu5090 'pgrep -f "policy_rollout.py paired record"' < /dev/null > /dev/null; do sleep 30; done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/policy_rollouts/$T/ results/program_b/policy_rollouts/$T/
scp -q gpu5090:wksp/stable_lyap_filters/results/jenga/$T.jsonl /tmp/remote_$T.jsonl
cat /tmp/remote_$T.jsonl >> results/jenga/$T.jsonl
LA_TAG=$T LA_WORKERS=30 python3 eval/lookahead_check.py truth > results/program_b/lookahead_truth.log 2>&1 || exit 1
LA_TAG=$T python3 eval/lookahead_check.py latent > results/program_b/lookahead_latent.log 2>&1 || exit 1
LA_TAG=$T python3 eval/lookahead_check.py report > results/program_b/lookahead_report.log 2>&1
