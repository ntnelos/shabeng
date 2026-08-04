"""
app.py — Native macOS Desktop Application wrapper for Shabeng
============================================================
"""
import sys
import os
import time
import threading
import uvicorn
import webview

# Ensure Homebrew and common binary paths are available when running as a .app bundle
os.environ["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}"

from server import app

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait briefly for server startup
    time.sleep(1.2)

    # Create native macOS webview window
    window = webview.create_window(
        title="Shabeng — Automated Wedding Reel & Audio Sync",
        url="http://127.0.0.1:8000",
        width=1280,
        height=850,
        resizable=True,
        min_size=(900, 650)
    )

    webview.start()
