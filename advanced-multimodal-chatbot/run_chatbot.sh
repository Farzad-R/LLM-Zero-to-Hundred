#!/bin/bash

# Start a new tmux session named "chatbot"
tmux new-session -d -s chatbot -n "RAG Reference"
tmux send-keys -t chatbot:0 "source ../../../python_env/whisper-env/bin/activate" Enter
tmux send-keys -t chatbot:0 "python src/utils/web_servers/rag_reference_service.py" Enter

tmux new-window -t chatbot -n "LLAVA"
tmux send-keys -t chatbot:1 "source ../../../python_env/whisper-env/bin/activate" Enter
tmux send-keys -t chatbot:1 "python src/utils/web_servers/llava_service.py" Enter

tmux new-window -t chatbot -n "SDXL"
tmux send-keys -t chatbot:2 "source ../../../python_env/whisper-env/bin/activate" Enter
tmux send-keys -t chatbot:2 "python src/utils/web_servers/sdxl_service.py" Enter

tmux new-window -t chatbot -n "STT"
tmux send-keys -t chatbot:3 "source ../../../python_env/whisper-env/bin/activate" Enter
tmux send-keys -t chatbot:3 "python src/utils/web_servers/stt_service.py" Enter

tmux new-window -t chatbot -n "App Server"
tmux send-keys -t chatbot:4 "source ../../../python_env/whisper-env/bin/activate" Enter
tmux send-keys -t chatbot:4 "python src/app.py" Enter

# Attach to the tmux session to view the terminals
tmux attach-session -t chatbot
