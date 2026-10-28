cd /home/corey/wksp/stable_lyap_filters
export T2_DIR=results/program_b/t2_v2
python3 -c "import sys;sys.path[:0]=['eval'];import t2_data as t;t.h(screen_workers=28, render_workers=5)" > results/program_b/t2_v2_h.log 2>&1 || exit 1
python3 -c "import sys;sys.path[:0]=['eval'];import t2_data as t;t.train_data(screen_workers=28, render_workers=5)" > results/program_b/t2_v2_train.log 2>&1
