#!/bin/zsh
# Ensure a single fresh tmux session named "lio" with 1 window split into 3 panes
tmux kill-session -t lio 2>/dev/null || true
tmux new-session -d -s lio -n lio

# Pane 0: localization
tmux send-keys -t lio:0.0 'cd nav/rosws/fastlio_localization && source devel/setup.zsh && roslaunch fast_lio_localization localization_mid360.launch' C-m
sleep 2

# Split pane 0 horizontally -> creates pane 1
tmux split-window -h -t lio:0.0
tmux send-keys -t lio:0.1 'cd livox_ros_driver2 && source devel/setup.zsh && roslaunch livox_ros_driver2 msg_MID360.launch' C-m
sleep 2

# Split the original left pane horizontally again -> creates pane 2 (three columns)
tmux split-window -h -t lio:0.0
tmux send-keys -t lio:0.2 'cd ~/teleoperation && python pos_server.py' C-m
sleep 1

# Arrange panes evenly and attach
tmux select-layout -t lio:0 even-horizontal
tmux attach-session -t lio