cd /home/corey/wksp/stable_lyap_filters
P=/home/corey/wksp/dp_venv/bin/python
$P eval/policy_rollout.py monitor 100 > results/program_b/policy_monitor.log 2>&1 &
$P eval/policy_rollout.py cem 100 > results/program_b/policy_cem.log 2>&1
wait
$P eval/policy_rollout.py abstain 100 > results/program_b/policy_abstain.log 2>&1
