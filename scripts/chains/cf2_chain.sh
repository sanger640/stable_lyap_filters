cd /home/corey/wksp/stable_lyap_filters
while kill -0 3978308 2>/dev/null; do sleep 30; done
/home/corey/wksp/dp_venv/bin/python eval/cf_rev.py run > results/jenga/cf2_run.log 2>&1
/home/corey/wksp/dp_venv/bin/python eval/cf_rev.py grade 2>&1 | grep -v -i warn
