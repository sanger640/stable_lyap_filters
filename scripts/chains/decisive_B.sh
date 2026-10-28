cd /home/corey/wksp/stable_lyap_filters
P=/home/corey/wksp/dp_venv/bin/python
export HARD_MASS=0.3 HARD_SHRINK=0
$P eval/policy_rollout.py paired monitor 100 > results/program_b/paired_monitor.log 2>&1 &
$P eval/policy_rollout.py paired cem 100 > results/program_b/paired_cem.log 2>&1
wait
