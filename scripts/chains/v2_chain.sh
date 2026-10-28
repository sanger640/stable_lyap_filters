cd /home/corey/wksp/stable_lyap_filters
until grep -q "filter_works_on_fork_states" results/program_b/filter_decisive_A.log; do sleep 30; done
CEM_VERSION=2 python3 eval/filter_states_cem.py > results/program_b/filter_decisive_A_v2.log 2>&1
while pgrep -f "^/home/corey/wksp/dp_venv/bin/python eval/policy_rollout.py paired cem" > /dev/null; do sleep 30; done
HARD_MASS=0.3 HARD_SHRINK=0 CEM_VERSION=2 /home/corey/wksp/dp_venv/bin/python eval/policy_rollout.py paired cem 100 > results/program_b/paired_cem_v2.log 2>&1
