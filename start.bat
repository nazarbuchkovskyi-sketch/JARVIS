@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Jarvis

if not exist "ui\dist\index.html" (
  echo [Jarvis] Building the interface...
  pushd ui
  call npm install --no-audit --no-fund || goto :fail
  call npm run build || goto :fail
  popd
)

if not exist "models\piper\ru_RU-ruslan-medium.onnx" (
  echo [Jarvis] Downloading the voice model...
  python -m jarvis.setup_models || goto :fail
)

python -m jarvis %*
goto :eof

:fail
echo [Jarvis] Start failed. See the messages above.
pause
