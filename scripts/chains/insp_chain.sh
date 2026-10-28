cd /home/corey/wksp/stable_lyap_filters
while pgrep -f "diag2.sh" > /dev/null; do sleep 30; done
HARD_MASS=0.3 INSP_WORKERS=28 python3 eval/inspector_data.py 20 > results/program_b/inspector_data.log 2>&1
