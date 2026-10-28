# latency test: each filter alone on the local RTX 3090, 5 dev episodes (15000-15004), closed loop, tag _lat
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
ENV="MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_lat EP_OFFSET=15000 SHARD=0/1 EXEC_NOISE_SCALE=2 HARD_OFF=1 CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1"
echo "lat start $(date)"
env $ENV $PY eval/irr_rollout.py lfck000 5 > logs/lat_lfck000.out 2>&1; echo "lfck000 done $(date)"
env $ENV $PY eval/policy_rollout.py paired lfc000 5 > logs/lat_lfc000.out 2>&1; echo "lfc000 done $(date)"
env $ENV $PY eval/policy_rollout.py paired lsfa030 5 > logs/lat_lsfa030.out 2>&1; echo "lsfa030 done $(date)"
