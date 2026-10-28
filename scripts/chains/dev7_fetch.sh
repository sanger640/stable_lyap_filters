cd /home/corey/wksp/stable_lyap_filters
until ! timeout 60 ssh rorqual 'squeue -u ali313 -h -n dev7 | grep -q .' < /dev/null; do sleep 60; done
for arm in record cbf cbfm; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev7; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
  rsync -a --include='ep*.json' --exclude='*' rorqual:/scratch/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev7 paired_cbf_dev7 paired_cbfm_dev7
python3 eval/cbf_dev7.py
