cd /home/corey/wksp/stable_lyap_filters
while pgrep -f "^/home/corey/wksp/dp_venv/bin/python eval/policy_rollout.py paired cem" > /dev/null; do sleep 30; done
python3 eval/filter_decisive_B.py > results/program_b/filter_decisive_B_final.log 2>&1
export HARD_MASS=0.3 LT_WORKERS=28
python3 eval/loop_truth.py results/program_b/policy_rollouts/paired_monitor > results/program_b/loop_truth_monitor.log 2>&1
python3 eval/loop_truth.py results/program_b/policy_rollouts/paired_cem_v2 > results/program_b/loop_truth_cem_v2.log 2>&1
python3 eval/loop_diagnose.py > results/program_b/loop_diagnose.log 2>&1
