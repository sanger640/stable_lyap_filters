cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/closedloop_train.log
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/closedloop_v1/train/ results/program_b/closedloop_v1/train/ >> $L 2>&1
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/closedloop_v1/val/ results/program_b/closedloop_v1/val/ >> $L 2>&1
python3 eval/closedloop_train.py modes >> $L 2>&1 || exit 1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 eval/closedloop_train.py us >> $L 2>&1 || exit 1
LA_TAG=paired_record_dev2x python3 eval/closedloop_train.py eval >> $L 2>&1
echo "chain done $(date)" >> $L
