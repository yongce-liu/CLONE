#!/bin/bash

# Start a new tmux session (if it doesn't already exist)
tmux new-session -d -s lio

# Window 1: Navigate to the localization directory and launch the ROS package
tmux send-keys -t lio 'cd nav/rosws/fastlio_localization && source devel/setup.bash && roslaunch fast_lio_localization localization_mid360.launch' C-m

# Wait for a while to ensure the first process starts
sleep 3

# Window 2: Navigate to the driver directory and launch the Livox ROS driver
tmux new-window -t lio:1 -n 'Livox ROS Driver'
tmux send-keys -t lio:1 'cd livox_ros_driver2 && source devel/setup.bash && roslaunch livox_ros_driver2 msg_MID360.launch' C-m

# Wait for a while to ensure the second process starts
sleep 3

# Window 3: Navigate to the teleoperation directory and run the Python server
tmux new-window -t lio:2 -n 'Teleoperation'
tmux send-keys -t lio:2 'cd ~/teleoperation && python pos_server.py' C-m

# Attach to the tmux session
tmux attach-session -t lio
