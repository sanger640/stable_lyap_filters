cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/closedloop_train.log
echo "restart $(date)" >> $L
for d in train p1train val; do
  rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/threeblock_v3/$d/ results/program_b/threeblock_v3/$d/ >> $L 2>&1 || exit 1
done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/threeblock_v3/val_recovery.json results/program_b/threeblock_v3/ >> $L 2>&1
echo "data copied $(date): $(ls results/program_b/threeblock_v3/train | wc -l) $(ls results/program_b/threeblock_v3/p1train | wc -l)" >> $L
python3 eval/closedloop_train.py modes >> $L 2>&1 || exit 1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 eval/closedloop_train.py us >> $L 2>&1 || exit 1
LA_TAG=paired_record_dev2x python3 eval/closedloop_train.py eval >> $L 2>&1
echo "chain done $(date)" >> $L
