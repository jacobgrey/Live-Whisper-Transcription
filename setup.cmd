@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Whisper Speech-to-Text - Setup
echo ============================================================
echo.

rem --- Self-locate ---
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

rem ============================================================
rem  Phase 0: Download project files if running standalone
rem ============================================================

if not exist "%ROOT%\src\whisper_daemon.py" (
    echo Project files not found. Downloading from GitHub...
    echo.

    set "REPO_URL=https://github.com/jacobgrey/Live-Whisper-Transcription/archive/refs/heads/main.zip"
    set "ZIP_FILE=%TEMP%\whisper-repo.zip"
    set "EXTRACT_DIR=%TEMP%\whisper-repo-extract"

    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '!REPO_URL!' -OutFile '!ZIP_FILE!' }" 2>nul

    if not exist "!ZIP_FILE!" (
        echo ERROR: Failed to download project files.
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )

    echo Extracting...
    powershell -Command "& { $ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path '!ZIP_FILE!' -DestinationPath '!EXTRACT_DIR!' -Force }" 2>nul

    rem The zip extracts to a subfolder named Live-Whisper-Transcription-main
    set "EXTRACTED="
    for /d %%D in ("!EXTRACT_DIR!\*") do set "EXTRACTED=%%D"

    if not defined EXTRACTED (
        echo ERROR: Extraction failed.
        del "!ZIP_FILE!" >nul 2>&1
        pause
        exit /b 1
    )

    rem Copy project files into current directory (where setup.cmd lives)
    xcopy "!EXTRACTED!\*" "%ROOT%\" /e /y /q >nul
    del "!ZIP_FILE!" >nul 2>&1
    rmdir /s /q "!EXTRACT_DIR!" >nul 2>&1

    if not exist "%ROOT%\src\whisper_daemon.py" (
        echo ERROR: Project files missing after extraction.
        pause
        exit /b 1
    )

    echo   Project files downloaded successfully.
    echo.
)

rem ============================================================
rem  Phase 1: Detect system and ask options
rem ============================================================

set "NEED_PYTHON=0"
set "NEED_FFMPEG=0"
set "NEED_AHK=0"
set "HAS_GPU=0"
set "INSTALL_DIARIZE=0"

rem --- Check Python 3.10 ---
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    set "NEED_PYTHON=1"
) else (
    for /f "tokens=*" %%V in ('py -3.10 --version 2^>^&1') do set "PY_VER=%%V"
)

rem --- Check ffmpeg ---
where ffmpeg >nul 2>&1
if errorlevel 1 set "NEED_FFMPEG=1"

rem --- Check AutoHotKey v2 ---
set "AHK_FOUND=0"
if exist "%ProgramFiles%\AutoHotkey\v2\AutoHotkey64.exe" set "AHK_FOUND=1"
if exist "%LocalAppData%\Programs\AutoHotkey\v2\AutoHotkey64.exe" set "AHK_FOUND=1"
where AutoHotkey64 >nul 2>&1 && set "AHK_FOUND=1"
if "!AHK_FOUND!"=="0" set "NEED_AHK=1"

rem --- Detect NVIDIA GPU ---
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "HAS_GPU=1"
        for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do set "GPU_NAME=%%G"
    )
)

rem --- Check if venv already exists ---
set "NEED_VENV=1"
if exist "%ROOT%\venv\Scripts\python.exe" set "NEED_VENV=0"

rem --- Ask about diarization ---
echo  Diarization adds speaker labels to transcriptions (Speaker 1,
echo  Speaker 2, etc.^) when transcribing files. It requires extra
echo  downloads (~2 GB^) and a free HuggingFace account. You can
echo  always add it later by re-running setup.
echo.
set /p "DIAR_CHOICE=Install diarization (speaker labels)? [y/N] "
if /i "!DIAR_CHOICE!"=="y" set "INSTALL_DIARIZE=1"
if /i "!DIAR_CHOICE!"=="yes" set "INSTALL_DIARIZE=1"
echo.

rem ============================================================
rem  Phase 2: Show the user what will happen
rem ============================================================

echo The following will be set up:
echo.

if "%NEED_PYTHON%"=="1" (
    echo  [INSTALL]  Python 3.10 (required - not found^)
) else (
    echo  [OK]       !PY_VER!
)

if "%NEED_FFMPEG%"=="1" (
    echo  [INSTALL]  ffmpeg (required - not found^)
) else (
    echo  [OK]       ffmpeg
)

if "%NEED_AHK%"=="1" (
    echo  [INSTALL]  AutoHotKey v2 (for push-to-talk hotkeys^)
) else (
    echo  [OK]       AutoHotKey v2
)

if "%NEED_VENV%"=="1" (
    echo  [INSTALL]  Python virtual environment + dependencies
) else (
    echo  [OK]       Virtual environment (already exists^)
)

if "%HAS_GPU%"=="1" (
    echo  [GPU]      !GPU_NAME! - will install CUDA-accelerated PyTorch
) else (
    echo  [CPU]      No NVIDIA GPU detected - will install CPU-only PyTorch
)

if "%INSTALL_DIARIZE%"=="1" (
    echo  [INSTALL]  Diarization (pyannote speaker labeling^)
) else (
    echo  [SKIP]     Diarization (not selected^)
)

echo.
set /p "CONFIRM=Proceed with installation? [Y/n] "
if /i "!CONFIRM!"=="n" (
    echo.
    echo Setup cancelled.
    pause
    exit /b 0
)
echo.

rem ============================================================
rem  Phase 3: Install prerequisites
rem ============================================================

rem --- Install Python 3.10 ---
if "%NEED_PYTHON%"=="1" (
    echo Downloading Python 3.10...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile '%TEMP%\python_setup.exe' }" 2>nul
    if not exist "%TEMP%\python_setup.exe" (
        echo ERROR: Failed to download Python installer.
        echo Install manually from: https://www.python.org/downloads/release/python-31011/
        pause
        exit /b 1
    )
    echo Installing Python 3.10 (this may take a minute^)...
    "%TEMP%\python_setup.exe" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
    if errorlevel 1 (
        echo ERROR: Python installation failed.
        del "%TEMP%\python_setup.exe" >nul 2>&1
        pause
        exit /b 1
    )
    del "%TEMP%\python_setup.exe" >nul 2>&1

    rem Refresh PATH so py launcher is available in this session
    set "PATH=%LocalAppData%\Programs\Python\Python310\;%LocalAppData%\Programs\Python\Python310\Scripts\;%PATH%"

    rem Verify installation
    py -3.10 --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3.10 installed but py launcher not found.
        echo You may need to restart your computer and re-run setup.
        pause
        exit /b 1
    )
    echo   Python 3.10 installed successfully.
    echo.
)

rem --- Install ffmpeg ---
if "%NEED_FFMPEG%"=="1" (
    echo Downloading ffmpeg...
    set "FFMPEG_DIR=%ROOT%\tools\ffmpeg"
    mkdir "!FFMPEG_DIR!" >nul 2>&1
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%TEMP%\ffmpeg.zip' }" 2>nul
    if not exist "%TEMP%\ffmpeg.zip" (
        echo ERROR: Failed to download ffmpeg.
        echo Install manually from: https://www.gyan.dev/ffmpeg/builds/
        pause
        exit /b 1
    )
    echo Extracting ffmpeg...
    powershell -Command "& { $ProgressPreference = 'SilentlyContinue'; Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%TEMP%\ffmpeg_extract' -Force; $bin = (Get-ChildItem -Path '%TEMP%\ffmpeg_extract' -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1).DirectoryName; Copy-Item -Path (Join-Path $bin '*') -Destination '!FFMPEG_DIR!' -Force }" 2>nul
    del "%TEMP%\ffmpeg.zip" >nul 2>&1
    rmdir /s /q "%TEMP%\ffmpeg_extract" >nul 2>&1

    if not exist "!FFMPEG_DIR!\ffmpeg.exe" (
        echo ERROR: ffmpeg extraction failed.
        echo Install manually from: https://www.gyan.dev/ffmpeg/builds/
        pause
        exit /b 1
    )

    rem Add to user PATH permanently
    powershell -Command "& { $current = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($current -notlike '*!FFMPEG_DIR!*') { [Environment]::SetEnvironmentVariable('Path', '!FFMPEG_DIR!;' + $current, 'User') } }" 2>nul
    set "PATH=!FFMPEG_DIR!;%PATH%"
    echo   ffmpeg installed to tools\ffmpeg and added to user PATH.
    echo.
)

rem --- Install AutoHotKey v2 ---
if "%NEED_AHK%"=="1" (
    echo Downloading AutoHotKey v2...
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/AutoHotkey/AutoHotkey/releases/download/v2.0.18/AutoHotkey_2.0.18_setup.exe' -OutFile '%TEMP%\ahk_setup.exe' }" 2>nul
    if exist "%TEMP%\ahk_setup.exe" (
        echo Installing AutoHotKey v2...
        "%TEMP%\ahk_setup.exe" /silent
        del "%TEMP%\ahk_setup.exe" >nul 2>&1
        echo   AutoHotKey v2 installed.
    ) else (
        echo   Could not download AutoHotKey. Install manually from:
        echo     https://www.autohotkey.com/
    )
    echo.
)

rem ============================================================
rem  Phase 4: Python environment + dependencies
rem ============================================================

rem --- Create venv ---
if "%NEED_VENV%"=="1" (
    echo Creating Python 3.10 virtual environment...
    py -3.10 -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists, reusing.
)
call "%ROOT%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
echo.

rem --- Install PyTorch ---
if "%HAS_GPU%"=="1" (
    echo Installing CUDA-enabled PyTorch (this may take several minutes^)...
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 ^
        --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo ERROR: PyTorch GPU installation failed.
        pause
        exit /b 1
    )
    (
        echo torch==2.7.1+cu118
        echo torchvision==0.22.1+cu118
        echo torchaudio==2.7.1+cu118
    ) > "%ROOT%\constraints.txt"
) else (
    echo Installing CPU-only PyTorch (this may take several minutes^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ERROR: PyTorch CPU installation failed.
        pause
        exit /b 1
    )
    pip freeze | findstr /i "^torch" > "%ROOT%\constraints.txt"
)
echo.

rem --- Install core dependencies ---
echo Installing core dependencies...
pip install -c "%ROOT%\constraints.txt" setuptools==70.3.0
pip install -c "%ROOT%\constraints.txt" huggingface_hub==0.22.2
pip install -c "%ROOT%\constraints.txt" faster-whisper==1.2.1 sounddevice soundfile numpy
echo.

rem --- Install diarization (optional) ---
if "%INSTALL_DIARIZE%"=="1" (
    echo Installing onnxruntime (CPU-only, before pyannote to prevent GPU variant^)...
    pip install -c "%ROOT%\constraints.txt" "onnxruntime==1.19.2"

    echo.
    echo Installing pyannote diarization stack...
    pip install -c "%ROOT%\constraints.txt" pyannote.audio==3.3.2
    pip install -c "%ROOT%\constraints.txt" matplotlib

    if "%HAS_GPU%"=="0" (
        echo.
        echo ============================================================
        echo  NOTE: No NVIDIA GPU detected.
        echo  Transcription will use CPU (slower but functional^).
        echo  Diarization (speaker labeling^) will be VERY SLOW on CPU.
        echo  Consider skipping it for long files.
        echo ============================================================
    )

    rem --- Verify onnxruntime ---
    echo.
    echo Verifying onnxruntime...
    python -c "import onnxruntime; print('  onnxruntime', onnxruntime.__version__, '- providers:', onnxruntime.get_available_providers())"

    rem --- Set PYANNOTE_MODEL env var ---
    echo.
    echo Setting PYANNOTE_MODEL environment variable...
    setx PYANNOTE_MODEL pyannote/speaker-diarization-3.1 >nul 2>&1

    rem --- HuggingFace token ---
    echo.
    if not exist "%ROOT%\hf_token.txt" (
        echo Diarization requires a HuggingFace API token.
        echo Get one at: https://huggingface.co/settings/tokens
        echo You must also accept the model license at:
        echo   https://huggingface.co/pyannote/speaker-diarization-3.1
        echo.
        set /p "HF_TOKEN=Paste your HF token (or press Enter to skip): "
        if defined HF_TOKEN (
            echo !HF_TOKEN!> "%ROOT%\hf_token.txt"
            echo   Token saved to hf_token.txt
        ) else (
            echo   Skipped. Create hf_token.txt manually later for diarization.
        )
    ) else (
        echo HuggingFace token file already exists.
    )
)

rem ============================================================
rem  Done
rem ============================================================

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To start:
echo    1. Run  scripts\start_daemon.cmd  (or press F7 in AHK^)
echo    2. Run  whisper-ptt.ahk  for push-to-talk hotkeys
echo    3. Drag files onto  scripts\Transcribe Drop.cmd  for batch
echo.
if "%HAS_GPU%"=="1" (
    echo  Mode: GPU (CUDA^)
) else (
    echo  Mode: CPU-only
)
if "%INSTALL_DIARIZE%"=="1" (
    echo  Diarization: installed
) else (
    echo  Diarization: not installed (re-run setup to add^)
)
echo ============================================================
pause
