cd /home/corey/wksp/stable_lyap_filters
ARMS="record cbfg lsfe010"
python3 cluster/merge_paired.py $(for a in $ARMS; do echo paired_${a}_h2h2; done)
AUDIT_OUT=results/jenga/outcome_audit_h2h2.json MUJOCO_GL=egl /home/corey/wksp/dp_venv/bin/python eval/outcome_audit.py runs $(for a in $ARMS; do echo paired_${a}_h2h2; done) > results/jenga/outcome_audit_h2h2.log 2>&1
H2H_STAGE=h2h2 python3 eval/h2h_analysis.py > results/jenga/lsf_h2h2_analysis.log 2>&1
echo done $(date)
