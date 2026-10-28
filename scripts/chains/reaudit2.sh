cd /home/corey/wksp/stable_lyap_filters
for d in paired_record_final2 paired_cbf_final2 paired_cbfm_final2 paired_record_dev7 paired_cbf_dev7 paired_cbfm_dev7; do
  mkdir -p results/program_b/policy_rollouts/$d
  timeout 1200 rsync -a --include='ep*.npz' --exclude='*' rorqual:/scratch/ali313/slf/repo/results/program_b/policy_rollouts/$d/ results/program_b/policy_rollouts/$d/
done
until ! pgrep -f "[o]utcome_audit.py runs" >/dev/null; do sleep 20; done
python3 eval/outcome_audit.py runs paired_record_final2 paired_cbf_final2 paired_cbfm_final2 paired_record_dev7 paired_cbf_dev7 paired_cbfm_dev7
echo REAUDIT2_DONE
