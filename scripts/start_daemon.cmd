@echo off
setlocal

set CUDA_MODULE_LOADING=LAZY
set PL_DISABLE_FABRIC=1
set PL_TORCH_DISTRIBUTED_BACKEND=gloo
set PYANNOTE_MODEL=pyannote/speaker-diarization-3.1

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "DAEMON=%ROOT%\src\whisper_daemon.py"

rem PyTorch ships its own CUDA/cuDNN DLLs here. Put this FIRST.
set "TORCH_LIB=%ROOT%\venv\Lib\site-packages\torch\lib"
set "PATH=%TORCH_LIB%;%PATH%"

rem Auto-detect CUDA toolkit bin (optional, PyTorch bundles its own CUDA libs)
set "CUDA_BIN="
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_BIN=%%~D\bin"
if defined CUDA_BIN set "PATH=%CUDA_BIN%;%PATH%"

"%PY%" "%DAEMON%"
