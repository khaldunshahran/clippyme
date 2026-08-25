@echo off
title ClippyMe Server
echo ========================================================
echo Starting ClippyMe Services...
echo ========================================================

cd /d "%~dp0"

echo [1/4] Ensuring Tailscale Serve is active...
start /b "" tailscale serve --bg 5175

echo [2/4] Starting Downloader Microservice (port 8001)...
start /b "" ".\.venv\Scripts\python.exe" -m uvicorn clippyme.services.downloader_api:app --host 127.0.0.1 --port 8001

echo [3/4] Starting Backend API (port 8000)...
start /b "" ".\.venv\Scripts\python.exe" -m uvicorn clippyme.api.app:app --host 127.0.0.1 --port 8000 --reload

echo [4/4] Starting Dashboard Frontend (port 5175)...
cd dashboard
start /b "" npm run dev

echo ========================================================
echo ClippyMe is running!
echo Local URL:     http://localhost:5175
echo Tailscale URL: https://mr-khaldun.tail9f387e.ts.net
echo ========================================================
echo You can minimize this window. Press any key to close this launcher.
pause >nul
