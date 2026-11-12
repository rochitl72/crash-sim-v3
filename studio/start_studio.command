#!/bin/bash
# Double-click to start CrashLab Studio (macOS). Opens http://127.0.0.1:8765 in your browser.
cd "$(dirname "$0")/.."
python3 studio/server.py
