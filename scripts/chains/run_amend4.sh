#!/bin/bash
cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/phase2_amend4.log
python3 eval/phase2_amend4.py readout >> $L 2>&1 || exit 1
python3 eval/phase2_amend4.py wide >> $L 2>&1
python3 eval/phase2_amend4.py long >> $L 2>&1
python3 eval/phase2_amend4.py grade >> $L 2>&1
