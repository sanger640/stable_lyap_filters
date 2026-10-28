cd /home/corey/wksp/stable_lyap_filters
until ! ssh nibi 'squeue -u ali313 -h -n dev4 | grep -q .' < /dev/null && ! timeout 60 ssh rorqual 'squeue -u ali313 -h -n dev4 | grep -q .' < /dev/null; do sleep 60; done
for arm in record cbf cbfopt; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev4; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
  rsync -a --include='ep*.json' --exclude='*' rorqual:/scratch/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev4 paired_cbf_dev4 paired_cbfopt_dev4
python3 eval/cbf_dev4.py
