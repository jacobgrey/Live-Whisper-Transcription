@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "SCRIPT=%ROOT%\src\transcribe_drop.py"

rem Auto-detect CUDA toolkit (optional)
set "CUDA_BIN="
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_BIN=%%~D\bin"
if defined CUDA_BIN set "PATH=%CUDA_BIN%;%PATH%"

"%PY%" "%SCRIPT%" %*
echo.
echo Done.
pause
