cd /home/corey/wksp/stable_lyap_filters
while pgrep -f "monitor_video.py sim" > /dev/null; do sleep 20; done
until grep -q "t2 h done" results/program_b/t2_v2_h.log; do sleep 20; done
python3 eval/monitor_video.py monitor > results/program_b/monitor_video_mon.log 2>&1 || exit 1
python3 eval/monitor_video.py compose >> results/program_b/monitor_video_mon.log 2>&1
