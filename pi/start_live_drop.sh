#!/usr/bin/env bash
# 桌面捷徑用：以系統 Python 啟動觸控版水滴偵測，輸出寫到 output/live_app.log
# 額外參數會傳給程式，例如 --fullscreen
cd "$(dirname "$0")" || exit 1
mkdir -p ../output
exec /usr/bin/python3 live_drop_detector.py "$@" >> ../output/live_app.log 2>&1
