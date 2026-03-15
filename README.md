# Live Whisper Transcription

Local speech-to-text for Windows using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). The Whisper model stays loaded in memory so transcription is near-instant after the first load.

**Features:**
- **Push-to-talk dictation** -- hold a hotkey, speak, release to paste transcribed text at your cursor
- **Batch transcription** -- drag audio files or folders onto a script to get text files
- **Speaker diarization** -- optionally label who said what (powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio))
- **GPU accelerated** -- uses NVIDIA CUDA when available, falls back to CPU automatically

## Quick Start

1. Download **[setup.cmd](https://raw.githubusercontent.com/jacobgrey/Live-Whisper-Transcription/main/setup.cmd)**
2. Put it in the folder where you want the app installed
3. Double-click it

That's it. The setup script automatically:
- Downloads the project files from GitHub
- Installs Python 3.10, ffmpeg, and AutoHotKey v2 if not already present
- Detects your GPU and installs the right version of PyTorch (CUDA or CPU-only)
- Creates a Python virtual environment with all dependencies
- Prompts for a HuggingFace token (needed for speaker diarization) if diarization is selected as an installed option.

Before anything is installed, you'll see a summary of what will be set up and can cancel.

> Alternatively, you can clone this repo and run `setup.cmd` from inside it.

## Usage

### Start the daemon

Run `scripts\start_daemon.cmd` (or press **F7** if the AHK script is running). The first launch takes about 15-30 seconds to load the model into memory. Subsequent transcriptions are fast.

### Push-to-talk (live dictation)

1. Run `whisper-ptt.ahk` (requires AutoHotKey v2)
2. **Hold F8** to record from your microphone
3. **Release F8** to transcribe and paste the text at your cursor
4. **Press F9** to shut down the daemon and free memory

### Batch transcription

Drag an audio file or folder onto `scripts\Transcribe Drop.cmd`. You'll be prompted to:
- Add speaker labels (diarization) if desired
- Include subfolders (for folder input)
- Mirror folder structure in output

Transcripts are saved as `.txt` files next to the source or in a `transcripts/` subfolder.

### Supported audio formats

Anything ffmpeg can decode: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.webm`, `.mp4`, and more.

## Speaker Diarization

Diarization identifies different speakers in the audio and labels them (Speaker 1, Speaker 2, etc.). To use it:

1. Get a [HuggingFace token](https://huggingface.co/settings/tokens)
2. Accept the [pyannote model license](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Either paste the token during setup, or save it to `hf_token.txt` in the project root

Diarization runs on GPU when available. On CPU it works but is significantly slower for long files.

## Architecture

A persistent daemon keeps the Whisper model loaded in memory and handles all transcription requests over a local TCP socket:

```
whisper-ptt.ahk  (push-to-talk)  ──┐
Transcribe Drop.cmd  (batch UI)  ──┼──► whisper_daemon.py  (TCP :8765)
whisper_client.py  (CLI)         ──┘         │
                                             ├─► faster-whisper  (transcription)
                                             ├─► sounddevice  (microphone)
                                             └─► diarize_worker.py  (subprocess)
                                                     └─► pyannote.audio  (speaker ID)
```

The diarization worker runs as an isolated subprocess to avoid CUDA DLL conflicts between faster-whisper (ctranslate2) and pyannote (PyTorch) on Windows.

## System Requirements

- **OS:** Windows 10 or 11
- **GPU (optional):** NVIDIA GPU with CUDA support (~2 GB VRAM for the Whisper model)
- **CPU mode:** Works on any machine, just slower

The setup script handles all other dependencies automatically.

## Troubleshooting

**Daemon won't start:** Make sure no other process is using port 8765. Check that `scripts\start_daemon.cmd` opens without errors.

**No text after releasing F8:** Verify the daemon is running (F7 to start it). Check that your microphone is set as the default recording device in Windows.

**Diarization fails:** Ensure `hf_token.txt` exists and contains a valid token. You must accept the model license on HuggingFace.

**Slow first transcription:** Normal -- the model takes 15-30 seconds to load on first start. After that, transcriptions are fast.

## License

MIT