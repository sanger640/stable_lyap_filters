cd /home/corey/wksp/stable_lyap_filters
until ! timeout 60 ssh rorqual 'squeue -u ali313 -h -n final2 | grep -q .' < /dev/null && ! timeout 60 ssh narval 'squeue -u ali313 -h -n final2 | grep -q .' < /dev/null; do sleep 120; done
for arm in record cbf cbfm; do
  d=results/program_b/policy_rollouts/paired_${arm}_final2; mkdir -p $d
  rsync -a --include='ep*.json' --exclude='*' rorqual:/scratch/ali313/slf/repo/$d/ $d/
  rsync -a --include='ep*.json' --exclude='*' narval:projects/def-soojeon/ali313/slf/repo/$d/ $d/
done
python3 cluster/merge_paired.py paired_record_final2 paired_cbf_final2 paired_cbfm_final2
python3 eval/cbf_final2.py
