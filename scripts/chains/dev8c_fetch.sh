cd /home/corey/wksp/stable_lyap_filters
until timeout 60 ssh nibi 'squeue -u ali313 -h -n dev8c | wc -l' < /dev/null | grep -qx 0; do sleep 60; done
for arm in record cbfmo3; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev8c; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev8c paired_cbfmo3_dev8c
python3 eval/cbf_dev8b.py
