#!/bin/bash
# Resumable Phase 1f data run: retry up to 4 times (per-chain screen files and per-state render files are kept).
cd /home/corey/wksp/stable_lyap_filters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3 4; do
  echo "=== attempt $attempt $(date)" >> results/program_b/phase1f_data.log
  python3 eval/phase1f_data.py --workers 5 >> results/program_b/phase1f_data.log 2>&1
  grep -q "^done" results/program_b/phase1f_data.log && break
  pkill -f "multiprocessing.spawn" -P $$ 2>/dev/null; sleep 20
done
grep -q "^done" results/program_b/phase1f_data.log && python3 eval/phase1f_models.py > results/program_b/phase1f_L.log 2>&1
