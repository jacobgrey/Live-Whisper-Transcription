import socket
import threading
import sys
import json
import re
import subprocess
import tempfile
from pathlib import Path
import shutil
import sounddevice as sd
import numpy as np
import soundfile as sf
from datetime import datetime
import time
import os

# ---------------------------------------------------------------------------
# Windows symlink workaround — non-admin users lack SeCreateSymbolicLinkPrivilege,
# which breaks huggingface_hub's cache (it relies on symlinks).  Patch os.symlink
# to fall back to copying so the model downloads work without elevated privileges.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _original_symlink = os.symlink

    def _symlink_or_copy(src, dst, target_is_directory=False, **kw):
        try:
            _original_symlink(src, dst, target_is_directory=target_is_directory, **kw)
        except OSError:
            src_path = Path(src) if not isinstance(src, Path) else src
            dst_path = Path(dst) if not isinstance(dst, Path) else dst
            # Resolve relative symlink targets against the destination's parent
            if not src_path.is_absolute():
                src_path = (dst_path.parent / src_path).resolve()
            if src_path.is_dir():
                shutil.copytree(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

    os.symlink = _symlink_or_copy

from faster_whisper import WhisperModel

HOST = "127.0.0.1"
PORT = 8765

MODEL_NAME = "distil-large-v3"
COMPUTE_TYPE = "int8_float16"
FOLDER_OUT_SUBDIR = "transcripts"
CHUNK_SECONDS = 15 * 60

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYANNOTE_MODEL = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")
DIARIZE_WORKER = Path(__file__).with_name("diarize_worker.py")

_language = "en"
_model = None
_recording = False
_audio = []
_stop = threading.Event()
_shutdown = threading.Event()
_input_device = None          # None = system default; int device index
job_lock = threading.Lock()

DEVICE_CONFIG = PROJECT_ROOT / "input_device.txt"


def _load_device_pref():
    """Load saved input device preference from input_device.txt."""
    global _input_device
    if DEVICE_CONFIG.exists():
        raw = DEVICE_CONFIG.read_text().strip()
        if raw == "" or raw.lower() == "default":
            _input_device = None
        else:
            try:
                _input_device = int(raw)
            except ValueError:
                _input_device = None
    else:
        _input_device = None


def _save_device_pref(device_index):
    """Persist the device preference to input_device.txt."""
    DEVICE_CONFIG.write_text(str(device_index) if device_index is not None else "default")


def _select_device_interactive():
    """Show input devices in the console and let the user pick one."""
    global _input_device
    devices = sd.query_devices()
    input_devs = []
    default_idx = sd.default.device[0]  # system default input device index
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " (system default)" if i == default_idx else ""
            input_devs.append((i, d['name'], marker))

    print("\n=== Input Device Selection ===")
    print("  0: System default")
    for idx, (dev_idx, name, marker) in enumerate(input_devs, 1):
        print(f"  {idx}: [{dev_idx}] {name}{marker}")

    current = _input_device
    if current is None:
        print(f"\nCurrently using: System default")
    else:
        print(f"\nCurrently using: device index {current}")

    print()
    while True:
        choice = input("Enter number (0 for system default, or press Enter to keep current): ").strip()
        if choice == "":
            break
        try:
            n = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if n == 0:
            _input_device = None
            _save_device_pref(None)
            print("Set to system default.\n")
            break
        if 1 <= n <= len(input_devs):
            dev_idx = input_devs[n - 1][0]
            dev_name = input_devs[n - 1][1]
            _input_device = dev_idx
            _save_device_pref(dev_idx)
            print(f"Set to: [{dev_idx}] {dev_name}\n")
            break
        print(f"Please enter 0-{len(input_devs)}.")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _noop(*_a, **_kw):
    """Default no-op progress callback used outside of socket jobs."""
    pass


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    elif m:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


def get_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe. ~100ms overhead, negligible."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    global _model
    log("Loading model...")
    try:
        _model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)
        log("Model loaded on GPU")
    except Exception as e:
        log(f"GPU load failed ({e}) — falling back to CPU")
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        log("Model loaded on CPU")


# ---------------------------------------------------------------------------
# Live microphone recording (hotkey path — unchanged, no progress needed)
# ---------------------------------------------------------------------------

def rec_loop():
    global _audio
    log("Mic stream opened")

    def cb(indata, frames, time_info, status):
        if _recording:
            _audio.append(indata.copy())

    with sd.InputStream(samplerate=16000, channels=1, callback=cb,
                        device=_input_device):
        _stop.wait()

    log("Mic stream closed")


def START():
    global _recording, _audio
    if _recording:
        return "ERR already"
    log("Mic recording started")
    _audio = []
    _stop.clear()
    _recording = True
    threading.Thread(target=rec_loop, daemon=True).start()
    return "OK"


def STOP():
    global _recording
    if not _recording:
        return "ERR not_recording"
    log("Mic recording stopped")
    _recording = False
    _stop.set()
    if not _audio:
        log("WARNING: No audio detected — mic may be muted or wrong device selected")
        return "OK "
    audio = np.concatenate(_audio, axis=0).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, 16000)
        t0 = time.time()
        seg, _ = _model.transcribe(f.name, language=_language, vad_filter=True)
        txt = " ".join(s.text.strip() for s in seg)
        log(f"Mic transcription done in {time.time() - t0:.2f}s")
        if not txt.strip():
            log("WARNING: No speech detected in recorded audio")
    return "OK " + txt


# ---------------------------------------------------------------------------
# Audio splitting
# ---------------------------------------------------------------------------

def split(src: Path, tmp: Path):
    log(f"Splitting {src.name}")
    out = tmp / "chunk_%03d.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-reset_timestamps", "1",
            str(out),
        ],
        check=True,
    )
    chunks = sorted(tmp.glob("chunk_*.wav"))
    log(f"{len(chunks)} chunk(s) created")
    return chunks


def _sanitize_out_name(p: Path) -> str:
    return re.sub(r"[^\w\-\. ]+", "_", p.stem) + ".txt"


def _extract_wav_segment(src_wav: Path, dst_wav: Path, start_s: float, end_s: float):
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
            "-i", str(src_wav),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst_wav),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Diarization via isolated subprocess (GPU-safe — no ctranslate2 DLL conflict)
# ---------------------------------------------------------------------------

def _load_hf_token() -> str | None:
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    try:
        token_file = PROJECT_ROOT / "hf_token.txt"
        if token_file.exists():
            return token_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _diarize_wav(wav_path: Path, use_cpu: bool = False) -> list[tuple[float, float, str]]:
    cmd = [sys.executable, str(DIARIZE_WORKER), str(wav_path)]
    if use_cpu:
        cmd.append("--cpu")

    env = {**os.environ}
    token = _load_hf_token()
    if token:
        env["HF_TOKEN"] = token

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, check=False,
            env=env, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("diarization worker timed out after 10 minutes")

    if result.returncode != 0:
        err_detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"diarize_worker exited {result.returncode}: {err_detail}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"diarize_worker returned non-JSON: {result.stdout[:200]}")

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"diarize_worker reported: {data['error']}")

    return [(s["start"], s["end"], s["speaker"]) for s in data]


# ---------------------------------------------------------------------------
# Batch transcription — plain
# ---------------------------------------------------------------------------

def transcribe_file(p: Path, out: Path, progress=_noop):
    log(f"File job started: {p.name}")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting audio{dur_str} ...")
        chunks = split(p, tmpdir)
        total = len(chunks)
        w = len(str(total))
        t0 = time.time()

        parts = []
        for idx, chunk in enumerate(chunks, 1):
            seg, _ = _model.transcribe(str(chunk), language=_language, vad_filter=True)
            parts.append(" ".join(s.text.strip() for s in seg))

            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx)
            progress(
                f"Chunk {idx:{w}}/{total} transcribed"
                f"  |  elapsed {fmt_elapsed(elapsed)}"
                f"  |  ETA {fmt_eta(eta)}"
            )

        txt = "\n\n".join(parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt + "\n", encoding="utf-8")
        log(f"File job finished in {fmt_elapsed(time.time() - t0)} -> {out.name}")

    finally:
        _cleanup_tmp(tmpdir)


# ---------------------------------------------------------------------------
# Batch transcription — diarized
# ---------------------------------------------------------------------------

def transcribe_file_diarized(p: Path, out: Path, progress=_noop):
    log(f"Diarized file job started: {p.name}")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_diar_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting audio{dur_str} ...")
        chunks = split(p, tmpdir)
        total = len(chunks)
        w = len(str(total))
        t0 = time.time()

        speaker_map: dict[str, str] = {}
        speaker_counter = 0
        merged_lines: list[str] = []
        last_speaker = None
        last_text_parts: list[str] = []

        def flush_last():
            nonlocal last_speaker, last_text_parts
            if last_speaker is None:
                return
            text = " ".join(x for x in last_text_parts if x).strip()
            if text:
                merged_lines.append(f"{last_speaker}: {text}")
            last_speaker = None
            last_text_parts = []

        for idx, chunk in enumerate(chunks, 1):
            elapsed_pre = time.time() - t0
            # ETA estimate: use pace of completed chunks; show "?" on first chunk
            if idx > 1:
                eta_str = f"  |  ETA ~{fmt_eta((elapsed_pre / (idx - 1)) * (total - idx + 1))}"
            else:
                eta_str = ""
            progress(
                f"Chunk {idx:{w}}/{total}  —  diarizing ..."
                f"  |  elapsed {fmt_elapsed(elapsed_pre)}{eta_str}"
            )

            segments = _diarize_wav(chunk)
            active = [s for s in segments if s[1] - s[0] >= 0.15]

            progress(
                f"Chunk {idx:{w}}/{total}  —  transcribing {len(active)} segment(s) ..."
            )

            for i, (st, en, spk) in enumerate(active):
                if spk not in speaker_map:
                    speaker_counter += 1
                    speaker_map[spk] = f"Speaker {speaker_counter}"

                label = speaker_map[spk]
                seg_wav = tmpdir / f"seg_{chunk.stem}_{i:04d}.wav"
                _extract_wav_segment(chunk, seg_wav, st, en)

                segs, _ = _model.transcribe(str(seg_wav), language=_language, vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()

                if not text:
                    continue

                if last_speaker is None:
                    last_speaker = label
                    last_text_parts = [text]
                elif last_speaker == label:
                    last_text_parts.append(text)
                else:
                    flush_last()
                    last_speaker = label
                    last_text_parts = [text]

            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx)
            progress(
                f"Chunk {idx:{w}}/{total}  done"
                f"  |  elapsed {fmt_elapsed(elapsed)}"
                f"  |  ETA {fmt_eta(eta)}"
            )

        flush_last()

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(merged_lines).strip() + "\n", encoding="utf-8")
        log(f"Diarized job finished in {fmt_elapsed(time.time() - t0)} -> {out.name}")

    finally:
        _cleanup_tmp(tmpdir)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup_tmp(tmpdir: Path):
    time.sleep(0.25)
    for f in tmpdir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    try:
        tmpdir.rmdir()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

def recv_line(c, maxb=32768):
    b = bytearray()
    while len(b) < maxb:
        x = c.recv(1)
        if not x or x == b"\n":
            break
        b += x
    return b.decode(errors="replace").strip()


def make_progress_sender(c):
    """Return a callable that streams a PROGRESS line to the connected client."""
    def send(msg: str):
        try:
            c.sendall(f"PROGRESS {msg}\n".encode("utf-8"))
        except Exception:
            pass
    return send


def client(c):
    try:
        cmd = recv_line(c)
        if not cmd:
            return

        parts = cmd.split(" ", 1)
        op = parts[0].upper()
        payload = parts[1] if len(parts) > 1 else ""

        progress = make_progress_sender(c)

        if op == "PING":
            resp = "OK pong"

        elif op == "START":
            resp = START()

        elif op == "STOP":
            with job_lock:
                resp = STOP()

        elif op in ("TRANSCRIBE_FILE", "TRANSCRIBE_FILE_DIARIZED"):
            d = json.loads(payload)
            p = Path(d["path"])
            out = p.with_name(_sanitize_out_name(p))

            with job_lock:
                if op == "TRANSCRIBE_FILE_DIARIZED" or d.get("diarize") is True:
                    transcribe_file_diarized(p, out, progress=progress)
                else:
                    transcribe_file(p, out, progress=progress)

            resp = f"OK {out}"

        elif op in ("TRANSCRIBE_FOLDER", "TRANSCRIBE_FOLDER_DIARIZED"):
            d = json.loads(payload)
            root = Path(d["path"])
            inc = d.get("include_subfolders", False)
            mir = d.get("mirror_structure", False)
            diarize = (op == "TRANSCRIBE_FOLDER_DIARIZED") or (d.get("diarize") is True)

            files = [p for p in (root.rglob("*") if inc else root.glob("*")) if p.is_file()]
            log(f"Folder job: {len(files)} file(s)")

            with job_lock:
                for file_idx, f in enumerate(files, 1):
                    progress(f"--- File {file_idx}/{len(files)}: {f.name} ---")

                    out = (
                        (root / FOLDER_OUT_SUBDIR / f.relative_to(root))
                        if mir
                        else (root / FOLDER_OUT_SUBDIR / f.name)
                    ).with_suffix(".txt")

                    if diarize:
                        transcribe_file_diarized(f, out, progress=progress)
                    else:
                        transcribe_file(f, out, progress=progress)

            resp = f"OK {root / FOLDER_OUT_SUBDIR}"

        elif op == "SHUTDOWN":
            log("Shutdown requested")
            _shutdown.set()
            resp = "OK bye"

        else:
            resp = "ERR unknown"

        c.sendall((resp + "\n").encode())

    except Exception as e:
        try:
            c.sendall((f"ERR {e}\n").encode())
        except Exception:
            pass

    finally:
        try:
            c.close()
        except Exception:
            pass


def main():
    _load_device_pref()

    if "--select-device" in sys.argv:
        _select_device_interactive()
    else:
        if _input_device is not None:
            dev_name = sd.query_devices(_input_device)['name']
            log(f"Input device: [{_input_device}] {dev_name}")
        else:
            log("Input device: system default")
        log("Tip: Launch with Shift+F7 to select a different input device.")

    load_model()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(16)
    log(f"Listening on {HOST}:{PORT}")

    while not _shutdown.is_set():
        try:
            srv.settimeout(1.0)
            c, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=client, args=(c,), daemon=True).start()

    srv.close()
    log("Daemon exiting")


if __name__ == "__main__":
    main()
