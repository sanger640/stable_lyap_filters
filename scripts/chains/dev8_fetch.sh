cd /home/corey/wksp/stable_lyap_filters
until timeout 60 ssh nibi 'squeue -u ali313 -h -n dev8 | wc -l' < /dev/null | grep -qx 0; do sleep 120; done
for arm in record cbfm cbfmo; do
  d=results/program_b/policy_rollouts/paired_${arm}_dev8; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' rorqual:/scratch/ali313/slf/repo/$d/ $d/
  rsync -a --include='ep*.json' --exclude='*' nibi:projects/def-soojeon/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_dev8 paired_cbfm_dev8 paired_cbfmo_dev8
python3 eval/cbf_dev8.py
