cd /home/corey/wksp/stable_lyap_filters
until [ $(ls results/program_b/policy_rollouts/paired_cbfrh_rb1/*.json | wc -l) -ge 200 ] && [ $(ls results/program_b/policy_rollouts/paired_cbfh_rb1/*.json | wc -l) -ge 200 ] && [ $(ls results/program_b/policy_rollouts/paired_record_rb1/*.json | wc -l) -ge 200 ]; do sleep 60; done
sleep 30
AUDIT_OUT=results/jenga/outcome_audit_rb1.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs paired_record_rb1 paired_cbfh_rb1 paired_cbfrh_rb1 > results/jenga/outcome_audit_rb1.log 2>&1
python3 eval/rbase_analysis.py > results/jenga/rbase_analysis.log 2>&1
echo done $(date)
