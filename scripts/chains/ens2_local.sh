cd /home/corey/wksp/stable_lyap_filters
export FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ENS_STEPS=4000 ENS_OUT=results/program_b/ens_v2
for pair in "0 1" "2 3"; do
  for k in $pair; do ENS_SEED=$k python3 eval/ens_train.py > results/program_b/ens2_$k.log 2>&1 & done
  wait
done
echo "ens2 done $(date)" >> results/program_b/ens2_0.log
