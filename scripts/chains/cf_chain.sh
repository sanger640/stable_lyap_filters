cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/cf_chain.log
echo "cf chain start $(date)" >> $L
until grep -q "^done" results/program_b/cf_data_local.log; do sleep 60; done
until ssh gpu5090 'grep -q "^done" ~/wksp/stable_lyap_filters/results/program_b/cf_data_remote.log' < /dev/null; do sleep 60; done
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/cf_v1/train/ results/program_b/cf_v1/train/ >> $L 2>&1
rsync -a gpu5090:wksp/stable_lyap_filters/results/program_b/cf_v1/val/ results/program_b/cf_v1/val/ >> $L 2>&1
echo "data: $(ls results/program_b/cf_v1/train | grep -c 'npz$') train, $(ls results/program_b/cf_v1/val | grep -c 'npz$') val $(date)" >> $L
FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 eval/cf_train.py >> $L 2>&1 || { echo "train failed" >> $L; exit 1; }
NEW=$PWD/results/program_b/cf_v1/model/model_CF.pt
OLD=$PWD/results/program_b/closedloop_v1/model/model_US.pt
echo "trained $(date)" >> $L
# 3a: the option checks on the same plans (truth cached)
CBF_WM=$NEW CHECKS_DIR=checks_cf python3 eval/cbf_qp_checks.py rescore >> $L 2>&1
CHECKS_DIR=checks_cf python3 eval/cbf_qp_checks.py report >> $L 2>&1
# 3b: barrier quality on held-out closed-loop decisions, old vs new
EVAL_OLD=$OLD EVAL_NEW=$NEW EVAL_OUT=results/jenga/cf_train_eval.json LA_TAG=paired_record_dev2x python3 eval/closedloop_train.py eval >> $L 2>&1
# 3c: the gradient CBF-QP again with the new model (offline, 211 dev decisions)
CBF_WM=$NEW QP_VERSION=1 QP_ITERS=10 QP_STEP_MM=10 QP_MAX_MM=40 python3 eval/cbf_qp.py opt cf_v1 >> $L 2>&1
CBF_WM=$NEW TRUTH_N=32 TRUTH_WORKERS=30 python3 eval/cbf_qp.py truth cf_v1 >> $L 2>&1
TRUTH_N=32 python3 eval/cbf_qp.py report cf_v1 >> $L 2>&1
echo "cf chain done $(date)" >> $L
