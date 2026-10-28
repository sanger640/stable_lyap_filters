#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# rendering encodes with DINOv2: wait until no other GPU job (P2-0b) is running
for attempt in 1 2 3 4; do
  echo "=== attempt $attempt $(date)" >> results/program_b/phase2_data.log
  while pgrep -f "phase2_p0.py" >/dev/null; do sleep 30; done
  python3 eval/phase2_data.py >> results/program_b/phase2_data.log 2>&1
  grep -q "^done" results/program_b/phase2_data.log && break
  sleep 20
done
