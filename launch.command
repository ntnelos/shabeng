#!/bin/bash
# Shabeng Desktop Launcher for macOS
# Double-click this file to run Shabeng and launch the Web UI automatically

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================"
echo "🎬 Starting Shabeng Web Application..."
echo "========================================"

# Launch browser after 1.5 seconds in background
(sleep 1.5 && open "http://localhost:8000") &

# Start server
python3 server.py
