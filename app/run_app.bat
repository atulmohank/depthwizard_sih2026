@echo off
REM DepthWizard web app -> http://127.0.0.1:8000   (Ctrl+C to stop)
cd /d "%~dp0\.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -c "import fastapi, uvicorn, multipart" 2>nul || "%PY%" -m pip install -r app\requirements.txt
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"
"%PY%" -m uvicorn app.server:app --host 127.0.0.1 --port 8000
