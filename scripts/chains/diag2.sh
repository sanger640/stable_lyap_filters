cd /home/corey/wksp/stable_lyap_filters
export HARD_MASS=0.3 LT_WORKERS=30 LT_SUBSET=1 LT_SAMPLE=500
python3 eval/loop_truth.py results/program_b/policy_rollouts/paired_monitor > results/program_b/loop_truth_monitor.log 2>&1
python3 eval/loop_truth.py results/program_b/policy_rollouts/paired_cem_v2 > results/program_b/loop_truth_cem_v2.log 2>&1
python3 eval/loop_diagnose.py > results/program_b/loop_diagnose.log 2>&1
