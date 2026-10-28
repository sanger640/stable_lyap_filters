#!/bin/bash
# SafeManip to Fir: code/assets/ckpt from the lab machine; the Rorqual-built venv + uv python + osmesa (same /scratch/ali313 paths) relayed.
S=scripts/chains/rq_relay
mkdir -p $S/libero $S/openpi_venv
ssh fir 'mkdir -p /scratch/ali313/safemanip /scratch/ali313/libero'
t0=$(date +%s)
( rsync -a rorqual:/scratch/ali313/libero/.uvpython rorqual:/scratch/ali313/libero/osmesa $S/libero/ && \
  rsync -a rorqual:/scratch/ali313/safemanip/SafeManip/eval/models/openpi/.venv/ $S/openpi_venv/ && echo "relay pulled $(( $(date +%s)-t0 ))s" && \
  rsync -a $S/libero/ fir:/scratch/ali313/libero/ && echo "relay pushed $(( $(date +%s)-t0 ))s" ) &
rsync -a --exclude=.venv --exclude='*.pyc' ~/wksp/safemanip/SafeManip ~/wksp/safemanip/robosuite ~/wksp/safemanip/ckpt fir:/scratch/ali313/safemanip/ && echo "local pushed $(( $(date +%s)-t0 ))s"
wait
rsync -a $S/openpi_venv/ fir:/scratch/ali313/safemanip/SafeManip/eval/models/openpi/.venv/ && echo "venv pushed $(( $(date +%s)-t0 ))s"
rsync -a ~/.cache/openpi/big_vision/paligemma_tokenizer.model fir:.cache/openpi/big_vision/ 2>/dev/null || ssh fir 'mkdir -p /scratch/ali313/openpi_cache/big_vision' && rsync -a ~/.cache/openpi/big_vision/paligemma_tokenizer.model fir:/scratch/ali313/openpi_cache/big_vision/
echo ALL_DONE
