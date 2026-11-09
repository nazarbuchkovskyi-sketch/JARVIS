@echo off
rem One-time setup on a new PC: Python packages, the window UI and the local models.
chcp 65001 >nul
cd /d "%~dp0"
title Jarvis setup

where python >nul 2>nul || (echo [Jarvis] Python 3.13 not found. Install it from python.org and tick "Add to PATH". & goto :fail)
where node >nul 2>nul || (echo [Jarvis] Node.js not found. Install the LTS version from nodejs.org. & goto :fail)
where claude >nul 2>nul || (echo [Jarvis] Claude Code not found. Install it: https://code.claude.com/docs - then run "claude" and log in. & goto :fail)

echo [Jarvis] 1/3 Python packages...
python -m pip install --upgrade pip || goto :fail
python -m pip install -r requirements.txt || goto :fail

echo [Jarvis] 2/3 Interface...
pushd ui
call npm ci --no-audit --no-fund || goto :fail
call npm run build || goto :fail
popd

echo [Jarvis] 3/3 Models (voice + speech recognition, about 900 MB)...
python -m jarvis.setup_models || goto :fail

echo.
echo [Jarvis] Done. Start Jarvis with start.bat
pause
goto :eof

:fail
echo [Jarvis] Setup failed. See the messages above.
pause
exit /b 1
