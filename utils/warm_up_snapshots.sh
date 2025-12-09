#!/bin/bash

# This script starts up the required servers in the background, then runs the snapshot warming utility.
# It blocks until the processes complete. If the user presses Ctrl+C, all servers are killed as well.

set -e

# Go up one directory (relative to the script's location)
cd "$(dirname "$0")/.."

# Function to kill all child processes
cleanup() {
    echo "Caught Ctrl+C. Killing all server processes..."
    kill $PID1 $PID2 $PID3 $PID4 2>/dev/null || true
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Start each server module in the background and save their PIDs
python -m app &
PID1=$!
python -m server.llm.sglang_server &
PID2=$!
python -m server.stt.parakeet_stt &
PID3=$!
python -m server.tts.kokoro_tts &
PID4=$!

# Run the snapshot warming utility (uncomment the next line to enable)
# python -m utils.warm_up_snapshots

# Wait for all servers; any exit/termination passes through
wait $PID1 $PID2 $PID3 $PID4

