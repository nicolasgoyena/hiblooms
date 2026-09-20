@echo off
REM Arranca backend (puerto 8000) y frontend (puerto 5173) en dos ventanas.
REM Doble clic desde la carpeta web\ o ejecútalo desde cualquier sitio.
set WEB=%~dp0
set REPO=%WEB%..

start "HIBLOOMS backend" cmd /k "cd /d %REPO% && python -m uvicorn web.backend.server:app --reload --port 8000"
start "HIBLOOMS frontend" cmd /k "cd /d %WEB%frontend && (if not exist node_modules npm install) && npm run dev -- --open"
