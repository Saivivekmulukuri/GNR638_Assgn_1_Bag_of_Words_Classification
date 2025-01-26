#!/bin/bash

# Name of the conda environment
ENV_NAME="gmflownet"

# Start a new tmux session
tmux new-session -s vocab_size_runs -d

# Commands for each vocab_size
tmux send-keys "conda activate $ENV_NAME && python3 proj3.py --vocab_size 100" C-m
tmux split-window -h "conda activate $ENV_NAME && python3 proj3.py --vocab_size 200"
tmux split-window -v "conda activate $ENV_NAME && python3 proj3.py --vocab_size 300"
tmux split-window -v "conda activate $ENV_NAME && python3 proj3.py --vocab_size 400"
tmux split-window -v "conda activate $ENV_NAME && python3 proj3.py --vocab_size 500"
tmux select-layout tiled

# Attach the session to monitor the outputs
tmux attach-session -t vocab_size_runs
