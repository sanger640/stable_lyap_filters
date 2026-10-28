cd /home/corey/wksp/stable_lyap_filters
until ! ssh nibi 'squeue -u ali313 -h -n dev5 | grep -q .' < /dev/null && ! timeout 60 ssh rorqual 'squeue -u ali313 -h -n dev5 | grep -q .' < /dev/null; do sleep 60; done
for arm in record cbf cbfq; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev5; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
  rsync -a --include='ep*.json' --exclude='*' rorqual:/scratch/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev5 paired_cbf_dev5 paired_cbfq_dev5
python3 eval/cbf_dev5.py
