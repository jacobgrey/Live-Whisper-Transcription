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

# Cosine-similarity threshold for matching a chunk-local speaker to a previously
# seen global speaker. Pyannote WeSpeaker embeddings typically score ~0.6-0.8 for
# same speaker and ~0.1-0.4 for different speakers; 0.55 balances both. Overridable.
SPEAKER_MATCH_THRESHOLD = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.55"))

_language = "en"
_model = None
_recording = False
_audio = []
_stop = threading.Event()
_shutdown = threading.Event()
_input_device = None          # None = system default; int device index
job_lock = threading.Lock()

DEVICE_CONFIG = PROJECT_ROOT / "config" / "input_device.txt"


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
        token_file = PROJECT_ROOT / "config" / "hf_token.txt"
        if token_file.exists():
            return token_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _diarize_wav(
    wav_path: Path,
    use_cpu: bool = False,
    speaker_hint: dict | None = None,
) -> tuple[list[tuple[float, float, str]], dict[str, np.ndarray]]:
    """Run diarization subprocess. Returns (segments, embeddings_by_local_label).

    `speaker_hint` may contain any of: num_speakers, min_speakers, max_speakers.
    `embeddings_by_local_label` maps each chunk-local SPEAKER_xx to its 1-D numpy
    embedding vector, or is empty if the pipeline didn't return embeddings.
    """
    cmd = [sys.executable, str(DIARIZE_WORKER), str(wav_path)]
    if use_cpu:
        cmd.append("--cpu")

    if speaker_hint:
        if speaker_hint.get("num_speakers") is not None:
            cmd += ["--num-speakers", str(int(speaker_hint["num_speakers"]))]
        else:
            if speaker_hint.get("min_speakers") is not None:
                cmd += ["--min-speakers", str(int(speaker_hint["min_speakers"]))]
            if speaker_hint.get("max_speakers") is not None:
                cmd += ["--max-speakers", str(int(speaker_hint["max_speakers"]))]

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

    # New format: {"segments": [...], "embeddings": {...}}.
    # Old format (pre-v2): [{"start": ..., "end": ..., "speaker": ...}, ...].
    if isinstance(data, list):
        segments = data
        embeddings_raw: dict = {}
    else:
        segments = data.get("segments", [])
        embeddings_raw = data.get("embeddings", {}) or {}

    seg_tuples = [(s["start"], s["end"], s["speaker"]) for s in segments]
    emb_by_label = {
        str(k): np.asarray(v, dtype=np.float32)
        for k, v in embeddings_raw.items()
        if v is not None
    }
    return seg_tuples, emb_by_label


# ---------------------------------------------------------------------------
# Cross-chunk speaker stitching
# ---------------------------------------------------------------------------

class GlobalSpeakerTracker:
    """Match chunk-local speaker labels to stable global speakers across chunks.

    Pyannote's SPEAKER_00/SPEAKER_01 labels are local to each pipeline call, so
    naive label reuse across chunks causes phantom speakers. This class keeps a
    running-mean embedding (centroid) per global speaker and matches new local
    speakers by cosine similarity.
    """

    def __init__(self, threshold: float = SPEAKER_MATCH_THRESHOLD):
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []     # index = global id
        self._counts: list[int] = []               # count of local speakers folded in
        self._names: list[str] = []                # "Speaker 1", "Speaker 2", ...
        self._fallback_next_id = 0                 # used only when no embeddings are available

    def _match(self, vec: np.ndarray) -> int | None:
        if not self._centroids:
            return None
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        best_idx, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            cn = np.linalg.norm(c)
            if cn == 0:
                continue
            sim = float(np.dot(vec, c) / (norm * cn))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx if best_sim >= self.threshold else None

    def _new_speaker(self, vec: np.ndarray | None) -> str:
        name = f"Speaker {len(self._names) + 1}"
        self._names.append(name)
        self._counts.append(1 if vec is not None else 0)
        self._centroids.append(vec.copy() if vec is not None else np.zeros(0, dtype=np.float32))
        return name

    def assign(self, local_label: str, embedding: np.ndarray | None) -> str:
        """Return the stable global name for a local chunk speaker."""
        if embedding is None or embedding.size == 0:
            # No embedding — fall back to treating each distinct local label as new.
            # This loses cross-chunk identity but preserves correctness within a chunk.
            name = f"Speaker {self._fallback_next_id + 1}"
            self._fallback_next_id += 1
            self._names.append(name)
            self._counts.append(0)
            self._centroids.append(np.zeros(0, dtype=np.float32))
            return name

        match = self._match(embedding)
        if match is None:
            return self._new_speaker(embedding)

        # Fold this embedding into the matched centroid (running mean).
        n = self._counts[match]
        c = self._centroids[match]
        self._centroids[match] = (c * n + embedding) / (n + 1)
        self._counts[match] = n + 1
        return self._names[match]


def _fmt_timestamp(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # negative or NaN
        seconds = 0.0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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

def transcribe_file_diarized(
    p: Path,
    out: Path,
    progress=_noop,
    speaker_hint: dict | None = None,
):
    log(f"Diarized file job started: {p.name}")
    if speaker_hint:
        log(f"Speaker hint: {speaker_hint}")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_diar_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting audio{dur_str} ...")
        chunks = split(p, tmpdir)
        total = len(chunks)
        w = len(str(total))
        t0 = time.time()

        tracker = GlobalSpeakerTracker()
        merged_lines: list[str] = []
        last_speaker: str | None = None
        last_text_parts: list[str] = []
        last_start_abs: float | None = None
        chunk_offset = 0.0  # seconds of source audio before the current chunk

        def flush_last():
            nonlocal last_speaker, last_text_parts, last_start_abs
            if last_speaker is None:
                return
            text = " ".join(x for x in last_text_parts if x).strip()
            if text:
                ts = _fmt_timestamp(last_start_abs or 0.0)
                merged_lines.append(f"[{ts}] {last_speaker}: {text}")
            last_speaker = None
            last_text_parts = []
            last_start_abs = None

        for idx, chunk in enumerate(chunks, 1):
            elapsed_pre = time.time() - t0
            if idx > 1:
                eta_str = f"  |  ETA ~{fmt_eta((elapsed_pre / (idx - 1)) * (total - idx + 1))}"
            else:
                eta_str = ""
            progress(
                f"Chunk {idx:{w}}/{total}  —  diarizing ..."
                f"  |  elapsed {fmt_elapsed(elapsed_pre)}{eta_str}"
            )

            segments, emb_by_label = _diarize_wav(chunk, speaker_hint=speaker_hint)
            active = [s for s in segments if s[1] - s[0] >= 0.15]

            # Resolve chunk-local labels to stable global names via embeddings.
            local_to_global: dict[str, str] = {}
            for _, _, spk in active:
                if spk in local_to_global:
                    continue
                local_to_global[spk] = tracker.assign(spk, emb_by_label.get(spk))

            progress(
                f"Chunk {idx:{w}}/{total}  —  transcribing {len(active)} segment(s) ..."
            )

            for i, (st, en, spk) in enumerate(active):
                label = local_to_global[spk]
                seg_wav = tmpdir / f"seg_{chunk.stem}_{i:04d}.wav"
                _extract_wav_segment(chunk, seg_wav, st, en)

                segs, _ = _model.transcribe(str(seg_wav), language=_language, vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()

                if not text:
                    continue

                seg_start_abs = chunk_offset + float(st)

                if last_speaker is None:
                    last_speaker = label
                    last_text_parts = [text]
                    last_start_abs = seg_start_abs
                elif last_speaker == label:
                    last_text_parts.append(text)
                else:
                    flush_last()
                    last_speaker = label
                    last_text_parts = [text]
                    last_start_abs = seg_start_abs

            # Advance chunk offset using the actual decoded chunk duration.
            chunk_offset += get_duration(chunk)

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
        log(
            f"Diarized job finished in {fmt_elapsed(time.time() - t0)} "
            f"-> {out.name}  ({len(tracker._names)} speaker(s))"
        )

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


def _extract_speaker_hint(d: dict) -> dict | None:
    """Pull num_speakers / min_speakers / max_speakers out of a socket payload."""
    hint: dict = {}
    for k in ("num_speakers", "min_speakers", "max_speakers"):
        v = d.get(k)
        if v is None:
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            hint[k] = n
    return hint or None


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
            hint = _extract_speaker_hint(d)

            with job_lock:
                if op == "TRANSCRIBE_FILE_DIARIZED" or d.get("diarize") is True:
                    transcribe_file_diarized(p, out, progress=progress, speaker_hint=hint)
                else:
                    transcribe_file(p, out, progress=progress)

            resp = f"OK {out}"

        elif op in ("TRANSCRIBE_FOLDER", "TRANSCRIBE_FOLDER_DIARIZED"):
            d = json.loads(payload)
            root = Path(d["path"])
            inc = d.get("include_subfolders", False)
            mir = d.get("mirror_structure", False)
            diarize = (op == "TRANSCRIBE_FOLDER_DIARIZED") or (d.get("diarize") is True)
            hint = _extract_speaker_hint(d) if diarize else None

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
                        transcribe_file_diarized(f, out, progress=progress, speaker_hint=hint)
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
