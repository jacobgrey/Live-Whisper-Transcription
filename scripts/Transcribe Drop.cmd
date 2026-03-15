@echo off
setlocal

set "ROOT=D:\Programs\Whisper Speech to Text\ptt"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "SCRIPT=%ROOT%\src\transcribe_drop.py"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

set "PATH=%CUDA_BIN%;%PATH%"

"%PY%" "%SCRIPT%" %*
echo.
echo Done.
pause