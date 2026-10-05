@echo off
REM Starts the backend (port 8000) and frontend (port 5173) in two windows.
REM Double-click from the web\ folder or run it from anywhere.
set WEB=%~dp0
set REPO=%WEB%..

start "HIBLOOMS backend" cmd /k "cd /d %REPO% && python -m uvicorn web.backend.server:app --reload --port 8000"
start "HIBLOOMS frontend" cmd /k "cd /d %WEB%frontend && (if not exist node_modules npm install) && npm run dev -- --open"
