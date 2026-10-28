#!/bin/bash
# Submit LSF training on Nibi once the clo upload and the cf data job are both done.
S=scripts/chains
P=$(cat $S/lsf_upload.pid); while kill -0 $P 2>/dev/null; do sleep 30; done
echo "upload done $(date)" >> $S/lsf_launch.log
while ssh nibi 'squeue -h -j 24697104' 2>/dev/null | grep -q .; do sleep 60; done
ssh nibi 'cd ~/projects/def-soojeon/ali313/slf/repo; ls results/program_b/lsf_v1/; cat results/program_b/lsf_v1/train_summary_cf.json 2>/dev/null' >> $S/lsf_launch.log 2>&1
if ssh nibi 'test -f ~/projects/def-soojeon/ali313/slf/repo/results/program_b/lsf_v1/train_summary_cf.json && test -f ~/projects/def-soojeon/ali313/slf/repo/results/program_b/lsf_v1/train_summary_clo.json'; then
  ssh nibi 'cd ~/projects/def-soojeon/ali313/slf/repo && sbatch cluster/lsf_train.sbatch' >> $S/lsf_launch.log 2>&1
else echo "MISSING PARTS, not submitted" >> $S/lsf_launch.log; fi
echo "launch step done $(date)" >> $S/lsf_launch.log
