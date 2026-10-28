cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/closedloop_train.log
echo "restart-remote $(date)" >> $L
ssh gpu5090 'pkill -f "policy_rollout.py paired cbf"' < /dev/null
rsync -a --relative eval results/program_b/closedloop_v1/selection.json results/program_b/closedloop_v1/train gpu5090:wksp/stable_lyap_filters/ >> $L 2>&1 || exit 1
echo "closed-loop data sent $(date)" >> $L
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && export FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && rm -rf results/program_b/closedloop_v1/model results/program_b/closedloop_v1/train_selected && python eval/closedloop_train.py modes && python eval/closedloop_train.py us' < /dev/null >> $L 2>&1 || exit 1
mkdir -p results/program_b/closedloop_v1/model
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/closedloop_v1/model/ results/program_b/closedloop_v1/model/ >> $L 2>&1
LA_TAG=paired_record_dev2x python3 eval/closedloop_train.py eval >> $L 2>&1
echo "chain done $(date)" >> $L
