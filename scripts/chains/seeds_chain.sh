cd /home/corey/wksp/stable_lyap_filters
for s in 1 2 3; do
  MC_SEED=$((20261100+s)) MC_OUT=seed$s WM_COMPILE=1 /home/corey/wksp/dp_venv/bin/python eval/margin_check.py run 0 1 > results/jenga/deep_dive/seed$s.log 2>&1
done
echo done $(date)
