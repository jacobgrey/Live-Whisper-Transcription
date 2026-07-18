# setup.ps1
# Whisper Speech-to-Text - Setup
#
# Transparent installer: shows exactly what it's going to do, asks for a
# single confirmation, then does it. No silent/passive installer flags.
#
# Only AutoHotkey v2 is installed system-wide (it has to run as a global
# hotkey daemon). Python and ffmpeg are installed entirely inside this
# project folder (venv\ and tools\ffmpeg\) via uv - nothing is added to
# the system PATH or registered machine-wide.
#
# Prerequisites you must already have:
# - uv (https://docs.astral.sh/uv/getting-started/installation/)
# - NVIDIA driver installed, if you want GPU acceleration

[CmdletBinding()]
param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# uv's `-c`/`--constraints` flag mishandles absolute paths that contain a
# space (confirmed independent of PowerShell - reproduces via cmd.exe too).
# Running from the project root and using paths relative to it for every
# uv invocation sidesteps the bug entirely, since none of those relative
# fragments (e.g. "config\constraints.txt") ever contain a space.
Set-Location -Path $Root

function Write-Banner($text) {
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host " $text"
    Write-Host ("=" * 60)
}

Write-Banner "Whisper Speech-to-Text - Setup"

if ($Rebuild) {
    if (Test-Path "venv") {
        Write-Host ""
        Write-Host "Rebuild requested: removing existing venv ($Root\venv)..."
        Remove-Item -Recurse -Force "venv"
    }
}

# ============================================================
# Phase 1: Detect prerequisites and current state
# ============================================================

# --- uv (hard prerequisite, not auto-installed) ---
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: 'uv' is required but was not found on PATH." -ForegroundColor Red
    Write-Host "Install it, then re-run this script:"
    Write-Host "  https://docs.astral.sh/uv/getting-started/installation/"
    Write-Host "  (PowerShell: irm https://astral.sh/uv/install.ps1 | iex)"
    exit 1
}

# --- AutoHotkey v2 (the one system-wide dependency) ---
function Test-Ahk2Installed {
    if (Test-Path "$env:ProgramFiles\AutoHotkey\v2\AutoHotkey64.exe") { return $true }
    if (Test-Path "$env:LocalAppData\Programs\AutoHotkey\v2\AutoHotkey64.exe") { return $true }
    if (Get-Command AutoHotkey64 -ErrorAction SilentlyContinue) { return $true }
    return $false
}
$needAhk = -not (Test-Ahk2Installed)

# --- NVIDIA GPU ---
$hasGpu = $false
$gpuName = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($gpuName) { $hasGpu = $true }
    } catch {}
}

# --- Existing project-local venv / ffmpeg ---
# Paths below are relative to $Root (see Set-Location above) so they never
# contain a space, regardless of the install folder's own name.
$VenvPath = "venv"
$VenvPy = "venv\Scripts\python.exe"
$needVenv = -not (Test-Path $VenvPy)

$FfmpegDir = "tools\ffmpeg"
$FfmpegExe = "tools\ffmpeg\ffmpeg.exe"
$needFfmpeg = -not (Test-Path $FfmpegExe)

# --- Ask about diarization ---
Write-Host ""
Write-Host "Diarization adds speaker labels to transcriptions (Speaker 1,"
Write-Host "Speaker 2, etc.) when transcribing files. It requires extra"
Write-Host "downloads (~2 GB) and a free HuggingFace account. You can"
Write-Host "always add it later by re-running setup."
Write-Host ""
$diarChoice = Read-Host "Install diarization (speaker labels)? [y/N]"
$installDiarize = $diarChoice -match '^(?i:y|yes)$'

# ============================================================
# Phase 2: Show the plan, ask once
# ============================================================

Write-Host ""
Write-Host "The following will be set up:"
Write-Host ""

if ($needAhk) {
    Write-Host " [INSTALL]  AutoHotKey v2 (for push-to-talk hotkeys) - SYSTEM-WIDE install"
} else {
    Write-Host " [OK]       AutoHotKey v2"
}

if ($needFfmpeg) {
    Write-Host " [INSTALL]  ffmpeg -> tools\ffmpeg  (project-local only)"
} else {
    Write-Host " [OK]       ffmpeg (project-local, already present)"
}

if ($needVenv) {
    Write-Host " [INSTALL]  Python 3.10 (via uv) + virtual environment (project-local)"
} else {
    Write-Host " [OK]       Virtual environment (already exists)"
}

if ($hasGpu) {
    Write-Host " [GPU]      $gpuName - will install CUDA-accelerated PyTorch"
} else {
    Write-Host " [CPU]      No NVIDIA GPU detected - will install CPU-only PyTorch"
}

if ($installDiarize) {
    Write-Host " [INSTALL]  Diarization (pyannote speaker labeling)"
} else {
    Write-Host " [SKIP]     Diarization (not selected)"
}

Write-Host ""
Write-Host "Nothing outside this project folder is touched except AutoHotkey v2."
Write-Host "Python and ffmpeg are installed locally into this project only -"
Write-Host "nothing is added to your system PATH."
Write-Host ""
$confirm = Read-Host "Proceed with installation? [Y/n]"
if ($confirm -match '^(?i:n|no)$') {
    Write-Host ""
    Write-Host "Setup cancelled."
    exit 0
}

# ============================================================
# Phase 3: Install
#
# Fast, unattended steps first; AutoHotkey (the one step requiring a
# manual click-through of its installer wizard) runs last so it doesn't
# block everything else behind a GUI window.
# ============================================================

# --- ffmpeg (project-local) ---
if ($needFfmpeg) {
    Write-Host ""
    Write-Host "Downloading ffmpeg into tools\ffmpeg (project-local, not added to system PATH)..."
    New-Item -ItemType Directory -Force -Path $FfmpegDir | Out-Null
    $ffmpegZip = Join-Path $env:TEMP "whisper-ffmpeg.zip"
    $ffmpegExtract = Join-Path $env:TEMP "whisper-ffmpeg-extract"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $ffmpegZip
    } catch {
        Write-Host "ERROR: Failed to download ffmpeg. Install manually from:" -ForegroundColor Red
        Write-Host "  https://www.gyan.dev/ffmpeg/builds/"
        exit 1
    }

    Write-Host "Extracting ffmpeg..."
    if (Test-Path $ffmpegExtract) { Remove-Item -Recurse -Force $ffmpegExtract }
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegExtract -Force
    $ffmpegBinDir = (Get-ChildItem -Path $ffmpegExtract -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1).DirectoryName

    if (-not $ffmpegBinDir) {
        Write-Host "ERROR: ffmpeg extraction failed." -ForegroundColor Red
        exit 1
    }

    Copy-Item -Path (Join-Path $ffmpegBinDir '*') -Destination $FfmpegDir -Force
    Remove-Item $ffmpegZip -Force -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $ffmpegExtract -ErrorAction SilentlyContinue

    if (-not (Test-Path $FfmpegExe)) {
        Write-Host "ERROR: ffmpeg not found after extraction." -ForegroundColor Red
        exit 1
    }
    Write-Host "  ffmpeg installed to tools\ffmpeg."
}

# --- Python 3.10 + venv, via uv (project-local) ---
Write-Host ""
Write-Host "Ensuring Python 3.10 is available (via uv)..."
& uv python install 3.10
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv failed to provision Python 3.10." -ForegroundColor Red
    exit 1
}

if ($needVenv) {
    Write-Host "Creating virtual environment (venv\)..."
    & uv venv --python 3.10 $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists, reusing."
}
Write-Host ""

# --- PyTorch (GPU or CPU variant) ---
$ConstraintsFile = "config\constraints.txt"
$RequirementsFile = "config\requirements.txt"
$RequirementsDiarizeFile = "config\requirements-diarize.txt"

$needTorch = $needVenv
if (-not $needTorch) {
    if ($hasGpu) {
        & $VenvPy -c "import torch; exit(0 if 'cu126' in torch.__version__ else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) { $needTorch = $true }
    } else {
        & $VenvPy -c "import torch; exit(0 if '+cu' not in torch.__version__ else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) { $needTorch = $true }
    }
}

if ($needTorch) {
    if ($hasGpu) {
        Write-Host "Installing CUDA-enabled PyTorch (this may take several minutes)..."
        & uv pip install --python $VenvPy torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: PyTorch GPU installation failed." -ForegroundColor Red; exit 1 }
        @(
            "torch==2.7.1+cu126"
            "torchvision==0.22.1+cu126"
            "torchaudio==2.7.1+cu126"
        ) | Set-Content -Path $ConstraintsFile -Encoding UTF8
    } else {
        Write-Host "Installing CPU-only PyTorch (this may take several minutes)..."
        & uv pip install --python $VenvPy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: PyTorch CPU installation failed." -ForegroundColor Red; exit 1 }
        & uv pip freeze --python $VenvPy | Select-String -Pattern '^torch' | ForEach-Object { $_.Line } | Set-Content -Path $ConstraintsFile -Encoding UTF8
    }
    Write-Host ""
} else {
    Write-Host "  PyTorch is up to date."
    Write-Host ""
}

# --- Core dependencies ---
$CoreMarker = "venv\.deps_core_installed"
$needCoreDeps = $true
if ((Test-Path $CoreMarker) -and -not $needTorch) {
    if ((Get-Content $RequirementsFile -Raw) -eq (Get-Content $CoreMarker -Raw)) {
        $needCoreDeps = $false
    }
}

if ($needCoreDeps) {
    Write-Host "Installing core dependencies..."
    & uv pip install --python $VenvPy -c $ConstraintsFile -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Core dependency installation failed." -ForegroundColor Red; exit 1 }
    Copy-Item -Path $RequirementsFile -Destination $CoreMarker -Force
    Write-Host ""
} else {
    Write-Host "  Core dependencies are up to date."
    Write-Host ""
}

# --- Diarization dependencies (optional) ---
if ($installDiarize) {
    $DiarMarker = "venv\.deps_diarize_installed"
    $needDiarDeps = $true
    if ((Test-Path $DiarMarker) -and -not $needTorch) {
        if ((Get-Content $RequirementsDiarizeFile -Raw) -eq (Get-Content $DiarMarker -Raw)) {
            $needDiarDeps = $false
        }
    }

    if ($needDiarDeps) {
        Write-Host "Installing diarization dependencies..."
        Write-Host "  (onnxruntime CPU-only installed before pyannote to prevent GPU variant)"
        & uv pip install --python $VenvPy -c $ConstraintsFile -r $RequirementsDiarizeFile
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Diarization dependency installation failed." -ForegroundColor Red; exit 1 }
        Copy-Item -Path $RequirementsDiarizeFile -Destination $DiarMarker -Force
        Write-Host ""
    } else {
        Write-Host "  Diarization dependencies are up to date."
        Write-Host ""
    }

    if (-not $hasGpu) {
        Write-Host "============================================================"
        Write-Host " NOTE: No NVIDIA GPU detected."
        Write-Host " Transcription will use CPU (slower but functional)."
        Write-Host " Diarization (speaker labeling) will be VERY SLOW on CPU."
        Write-Host " Consider skipping it for long files."
        Write-Host "============================================================"
        Write-Host ""
    }

    Write-Host "Verifying onnxruntime..."
    & $VenvPy -c "import onnxruntime; print('  onnxruntime', onnxruntime.__version__, '- providers:', onnxruntime.get_available_providers())"
    Write-Host ""

    Write-Host "Setting PYANNOTE_MODEL environment variable..."
    setx PYANNOTE_MODEL pyannote/speaker-diarization-3.1 | Out-Null
    Write-Host ""

    $HfTokenFile = "config\hf_token.txt"
    if (-not (Test-Path $HfTokenFile)) {
        Write-Host "Diarization requires a HuggingFace API token."
        Write-Host "Get one at: https://huggingface.co/settings/tokens"
        Write-Host "You must also accept the model license at:"
        Write-Host "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        Write-Host ""
        $hfToken = Read-Host "Paste your HF token (or press Enter to skip)"
        if ($hfToken) {
            Set-Content -Path $HfTokenFile -Value $hfToken -Encoding UTF8
            Write-Host "  Token saved to config\hf_token.txt"
        } else {
            Write-Host "  Skipped. Create config\hf_token.txt manually later for diarization."
        }
    } else {
        Write-Host "HuggingFace token file already exists."
    }
}

# --- AutoHotkey v2 (last: the one step needing a manual install wizard) ---
if ($needAhk) {
    Write-Host ""
    Write-Host "Downloading AutoHotkey v2..."
    $ahkUrl = "https://github.com/AutoHotkey/AutoHotkey/releases/download/v2.0.18/AutoHotkey_2.0.18_setup.exe"
    $ahkInstaller = Join-Path $env:TEMP "ahk_setup.exe"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $ahkUrl -OutFile $ahkInstaller
    } catch {
        $ahkInstaller = $null
    }
    if ($ahkInstaller -and (Test-Path $ahkInstaller)) {
        Write-Host "Launching AutoHotkey v2 installer - complete the install wizard to continue..."
        Start-Process -FilePath $ahkInstaller -Wait
        Remove-Item $ahkInstaller -Force -ErrorAction SilentlyContinue

        if (Test-Ahk2Installed) {
            Write-Host "  AutoHotkey v2 installed."
        } else {
            Write-Host "  AutoHotkey v2 install wasn't detected after the wizard closed. You may need to install it manually:" -ForegroundColor Yellow
            Write-Host "    https://www.autohotkey.com/"
        }
    } else {
        Write-Host "  Could not download AutoHotkey. Install manually from:" -ForegroundColor Yellow
        Write-Host "    https://www.autohotkey.com/"
    }
}

# ============================================================
# Done
# ============================================================

Write-Banner "Setup complete!"
Write-Host ""
Write-Host " To start:"
Write-Host "   1. Run  scripts\start_daemon.cmd  (or press F7 in AHK)"
Write-Host "   2. Run  whisper-ptt.ahk  for push-to-talk hotkeys"
Write-Host "   3. Drag files onto  scripts\Transcribe Drop.cmd  for batch"
Write-Host ""
if ($hasGpu) {
    Write-Host " Mode: GPU (CUDA)"
} else {
    Write-Host " Mode: CPU-only"
}
if ($installDiarize) {
    Write-Host " Diarization: installed"
} else {
    Write-Host " Diarization: not installed (re-run setup to add)"
}
Write-Host "============================================================"
