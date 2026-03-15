# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Windows GPU-accelerated local speech-to-text system using a **daemon/client architecture**. The Whisper model stays loaded in VRAM for fast repeated transcription. Supports push-to-talk live dictation (via AutoHotKey) and batch file/folder transcription with optional speaker diarization.

## Architecture

```
User Interfaces                          Daemon Server
─────────────────                        ──────────────
whisper-ptt.ahk (F8 push-to-talk)  ─┐
transcribe_drop.py (drag-drop UI)   ─┼─► whisper_daemon.py (TCP 127.0.0.1:8765)
whisper_client.py (CLI)             ─┘        │
                                              ├─► faster-whisper (GPU transcription)
                                              ├─► sounddevice (mic recording)
                                              └─► diarize_worker.py (subprocess)
                                                      └─► pyannote.audio (speaker ID)
```

**Why diarize_worker.py is a subprocess:** Running pyannote in-process with ctranslate2 (used by faster-whisper) causes Windows CUDA DLL symbol conflicts. The subprocess isolation solves this.

## Folder Structure

```
ptt/
├── src/                        # Python source code
│   ├── whisper_daemon.py       # Socket server (main daemon)
│   ├── whisper_client.py       # Minimal TCP client (CLI)
│   ├── transcribe_drop.py      # Batch drag-drop UI
│   └── diarize_worker.py       # Isolated diarization subprocess
├── scripts/                    # Windows batch scripts
│   ├── start_daemon.cmd        # Launches daemon with CUDA env setup
│   ├── Transcribe Drop.cmd     # Drag-drop wrapper for transcribe_drop.py
│   └── rebuild_whisper_env.cmd # Recreates venv with pinned deps
├── config/                     # Configuration and dependency files
│   ├── requirements.txt        # Core Python dependencies
│   ├── requirements-diarize.txt # Diarization dependencies
│   ├── constraints.txt         # Torch version pins (generated, gitignored)
│   └── hf_token.txt            # HF API token (gitignored)
├── setup.cmd                   # One-click installer (detects GPU, installs everything)
├── whisper-ptt.ahk             # AutoHotKey v2.0+ hotkeys (F7/F8/F9)
└── venv/                       # Python 3.10 virtual environment (gitignored)
```

## Key Files

- **src/whisper_daemon.py** — Socket server; loads distil-large-v3 model, handles START/STOP (mic), TRANSCRIBE_FILE, TRANSCRIBE_FOLDER commands. Splits long audio into 15-min chunks via ffmpeg. Uses `PROJECT_ROOT` constant for paths outside `src/`.
- **src/diarize_worker.py** — Isolated subprocess for speaker diarization. Takes a WAV path, outputs JSON speaker segments. **No ctranslate2 imports allowed here.**
- **src/transcribe_drop.py** — User-facing batch interface. Prompts for diarization, subfolders, structure mirroring. Streams progress from daemon.
- **src/whisper_client.py** — Minimal TCP client for sending commands to daemon.
- **whisper-ptt.ahk** — AutoHotKey v2.0+ hotkeys: F8 (hold=record, release=transcribe+paste), F7 (start daemon), F9 (shutdown). Uses `A_ScriptDir` for portable paths.
- **scripts/start_daemon.cmd** — Launches daemon with CUDA env vars and PyTorch DLL path setup.
- **scripts/rebuild_whisper_env.cmd** — Recreates the Python 3.10 venv with all pinned dependencies.

## Socket Protocol

Line-delimited text over TCP. Commands: `PING`, `START`, `STOP`, `TRANSCRIBE_FILE <json>`, `TRANSCRIBE_FILE_DIARIZED <json>`, `TRANSCRIBE_FOLDER <json>`, `TRANSCRIBE_FOLDER_DIARIZED <json>`, `SHUTDOWN`. Responses are `PROGRESS ...\n` lines followed by `OK result\n` or `ERR message\n`.

## Installation

Run `setup.cmd` (double-click or from terminal). It will:
1. Verify Python 3.10 and ffmpeg are installed
2. Detect NVIDIA GPU — installs CUDA PyTorch if found, CPU-only otherwise
3. Create venv and install all dependencies
4. Prompt for HuggingFace token (needed for diarization)
5. Install AutoHotKey v2 if not found

Prerequisites the user must install manually: **Python 3.10** and **ffmpeg** (must be in PATH).

All paths are portable — scripts auto-detect their location via `%~dp0` (cmd) and `A_ScriptDir` (AHK). CUDA toolkit is auto-detected from standard install locations.

## Running the System

```bash
# Start daemon (keeps model in VRAM)
scripts/start_daemon.cmd

# Test daemon is alive
python src/whisper_client.py PING

# Shutdown daemon (frees VRAM)
python src/whisper_client.py SHUTDOWN

# Rebuild venv from scratch
scripts/rebuild_whisper_env.cmd
```

All Python commands should use the venv at `./venv/`. The `start_daemon.cmd` script handles venv activation and CUDA environment setup.

## Critical Dependency Constraints

- **setuptools==70.3.0** — Pinned; newer versions removed `pkg_resources` which pyannote needs.
- **torch/torchvision/torchaudio** — GPU installs use matching CUDA 11.8 trio (`+cu118` wheels). CPU installs use standard PyPI wheels.
- **onnxruntime==1.19.2** — CPU-only; prevents GPU variant from causing conflicts.
- **PyTorch 2.6+** — Requires `weights_only=False` workaround for checkpoint loading in diarize_worker.
- **PL_DISABLE_FABRIC=1** — Set in start_daemon.cmd to avoid PyTorch Lightning import hangs.

## Hardware

- GPU mode: NVIDIA GPU with CUDA support required. Whisper model uses ~2 GB VRAM; diarization uses ~1-2 GB per subprocess.
- CPU mode: Works on any machine but transcription is slower and diarization can be very slow on long files.

## Diarization Design

Speaker labels are tracked across 15-min chunks using a global `speaker_map` dict. Each chunk is diarized independently by `diarize_worker.py`, then segments are mapped to consistent "Speaker 1", "Speaker 2" labels. Segments shorter than 0.15s are filtered as noise. Consecutive same-speaker segments are merged.

## HF Token

Required for pyannote model downloads. Read from `config/hf_token.txt` or environment variables (`HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`).