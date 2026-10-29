# H2H5 deck videos (descriptive): 3 episodes x 4 arms at 2x, replayed from the saved decision states (no world model, no GPU pool)
cd /home/corey/wksp/stable_lyap_filters
NO_SLICE=1 NPROC=4 nice -n 10 /home/corey/wksp/dp_venv/bin/python eval/filter_videos.py H5
