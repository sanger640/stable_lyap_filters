#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
exec python3 eval/phase1_data.py --workers 4 > results/program_b/phase1_data.log 2>&1
