#!/bin/zsh
# A robust tmux launcher for Fast-LIO localization setup

SESSION=lio
WINDOW=lio

# --- Clean up any existing session ---
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Killing existing tmux session: $SESSION"
  tmux kill-session -t $SESSION
  sleep 1
fi

# --- Start a new detached session with one window ---
echo "Starting tmux session: $SESSION"
tmux new-session -d -s $SESSION -n $WINDOW

# --- Pane 0: localization ---
echo "Launching localization..."
tmux send-keys -t ${SESSION}:0.0 'source devel/setup.zsh && roslaunch fast_lio_localization localization_mid360.launch' C-m
sleep 2

# --- Pane 1: livox driver (split horizontally from left pane) ---
echo "Launching livox driver..."
tmux select-pane -t ${SESSION}:0.0
tmux split-window -h -t ${SESSION}:0
sleep 0.5
tmux send-keys -t ${SESSION}:0.1 'source devel/setup.zsh && roslaunch livox_ros_driver2 msg_MID360.launch' C-m
sleep 2

# --- Pane 2: pos_server (split horizontally again from the RIGHT pane) ---
echo "Launching pos_server..."
tmux select-pane -t ${SESSION}:0.1
tmux split-window -h -t ${SESSION}:0
sleep 0.5
tmux send-keys -t ${SESSION}:0.2 'source devel/setup.zsh && python /catkin_ws/deploy_onboard/pos_server.py' C-m
sleep 1

# --- Arrange evenly and attach ---
tmux select-layout -t ${SESSION}:0 even-horizontal
tmux select-pane -t ${SESSION}:0.0

echo "All panes launched. Attaching..."
tmux attach-session -t $SESSION
