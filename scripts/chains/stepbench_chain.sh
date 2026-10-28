cd /home/corey/wksp/stable_lyap_filters
until grep -qsE "^\{.cases|Traceback" results/jenga/deep_dive/prune_check.log && ! pgrep -f "[p]rune_check.py" >/dev/null; do sleep 60; done
/home/corey/wksp/dp_venv/bin/python eval/step_bench.py run > results/jenga/deep_dive/stepbench.log 2>&1
/home/corey/wksp/dp_venv/bin/python eval/step_bench.py analyse > results/jenga/deep_dive/stepbench_analyse.log 2>&1
cat results/jenga/deep_dive/stepbench_analyse.log
