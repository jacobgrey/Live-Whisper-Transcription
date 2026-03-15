@echo off
setlocal

set CUDA_MODULE_LOADING=LAZY
set PL_DISABLE_FABRIC=1
set PL_TORCH_DISTRIBUTED_BACKEND=gloo
set PYANNOTE_MODEL=pyannote/speaker-diarization-3.1

set "ROOT=D:\Programs\Whisper Speech to Text\ptt"
set "PY=%ROOT%\venv\Scripts\python.exe"
set "DAEMON=%ROOT%\src\whisper_daemon.py"

rem PyTorch ships its own CUDA/cuDNN DLLs here. Put this FIRST.
set "TORCH_LIB=%ROOT%\venv\Lib\site-packages\torch\lib"

rem Keep your CUDA toolkit bin available, but AFTER torch\lib
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

set "PATH=%TORCH_LIB%;%CUDA_BIN%;%PATH%"

"%PY%" "%DAEMON%"