@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Jarvis

python -c "import fastapi, onnx_asr, piper, silero_vad, sounddevice, cv2" >nul 2>nul || (
  echo [Jarvis] Python packages are missing. Run setup.bat first.
  goto :fail
)

if not exist "ui\dist\index.html" (
  echo [Jarvis] Building the interface...
  pushd ui
  call npm ci --no-audit --no-fund || goto :fail
  call npm run build || goto :fail
  popd
)

if not exist "models\.ready" (
  echo [Jarvis] Downloading the models, once - about 900 MB...
  python -m jarvis.setup_models || goto :fail
)

python -m jarvis %*
goto :eof

:fail
echo [Jarvis] Start failed. See the messages above.
pause
