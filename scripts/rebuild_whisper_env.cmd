@echo off
setlocal

echo ==========================================
echo Rebuilding Whisper + Diarization Environment
echo ==========================================

cd /d "%~dp0.."

echo.
echo Removing existing venv...
if exist venv rmdir /s /q venv

echo.
echo Creating fresh venv (Python 3.10 required)...
py -3.10 -m venv venv

call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing CUDA 12.6 Torch stack...
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 ^
  --index-url https://download.pytorch.org/whl/cu126

echo.
echo Creating torch constraints file...
echo torch==2.7.1+cu126 > config\constraints.txt
echo torchvision==0.22.1+cu126 >> config\constraints.txt
echo torchaudio==2.7.1+cu126 >> config\constraints.txt

echo.
echo Installing core dependencies...
pip install -c config\constraints.txt -r config\requirements.txt
copy /y config\requirements.txt venv\.deps_core_installed >nul

echo.
echo --- FIX: Pin CPU-only onnxruntime BEFORE pyannote to prevent onnxruntime-gpu ---
echo --- being pulled in as a transitive dep, which causes cuDNN symbol conflicts.  ---
echo --- PyTorch already handles GPU for both Whisper and pyannote; onnxruntime     ---
echo --- is only used for minor preprocessing steps and CPU is fine there.          ---
echo.
echo Installing diarization dependencies...
pip install -c config\constraints.txt -r config\requirements-diarize.txt
copy /y config\requirements-diarize.txt venv\.deps_diarize_installed >nul

echo.
echo Verifying onnxruntime was not upgraded to GPU variant...
python -c "import onnxruntime; print('onnxruntime version:', onnxruntime.__version__); providers = onnxruntime.get_available_providers(); print('providers:', providers)"

echo.
echo Setting diarization model to stable 3.1...
setx PYANNOTE_MODEL pyannote/speaker-diarization-3.1

echo.
echo ==========================================
echo Build complete.
echo Restart daemon after this.
echo ==========================================
pause