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

# Audio cleanup filter chain:
# - highpass=80  : cut rumble, HVAC, mic handling (below voice fundamental)
# - acompressor  : gentle 2:1 compression to even within-speaker dynamics
# - loudnorm     : normalize to -16 LUFS so pyannote embeddings don't drift
#                  when a speaker is sometimes loud and sometimes quiet
AUDIO_CLEANUP_FILTER = (
    "highpass=f=80,"
    "acompressor=threshold=-18dB:ratio=2:attack=5:release=50,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)

# Silence-aware chunking: detect silence >= MIN_SILENCE_SEC below NOISE_DB,
# snap each 15-minute target boundary to the nearest silence mid-point within
# BOUNDARY_WINDOW_SEC. Prevents word-straddling at chunk seams.
SILENCE_NOISE_DB = int(os.environ.get("SILENCE_NOISE_DB", "-30"))
SILENCE_MIN_SEC = float(os.environ.get("SILENCE_MIN_SEC", "0.5"))
BOUNDARY_WINDOW_SEC = float(os.environ.get("BOUNDARY_WINDOW_SEC", "30"))

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

def _find_silence_regions(src: Path) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect on the raw input and return (start, end) pairs.

    For multi-stream files (e.g. OBS mic + Discord), mixes all audio streams
    before detecting so a boundary only lands in silence when *every* stream
    is quiet — otherwise a silent mic while Discord talks would create bad cuts.
    """
    silence_filter = (
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:duration={SILENCE_MIN_SEC}"
    )
    streams = probe_audio_streams(src)
    if len(streams) >= 2:
        inputs = "".join(f"[0:a:{i}]" for i in range(len(streams)))
        fc = f"{inputs}amix=inputs={len(streams)}:normalize=0[m];[m]{silence_filter}"
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(src),
            "-filter_complex", fc,
            "-f", "null", "-",
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(src),
            "-af", silence_filter,
            "-f", "null", "-",
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    regions: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            cur_start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur_start is not None:
            regions.append((cur_start, float(m.group(1))))
            cur_start = None
    return regions


def _compute_silence_boundaries(
    total_duration: float,
    silences: list[tuple[float, float]],
    target_chunk_sec: float = CHUNK_SECONDS,
    window_sec: float = BOUNDARY_WINDOW_SEC,
) -> list[float]:
    """For each target boundary (target_chunk_sec * k), snap to the nearest
    silence mid-point within +/- window_sec. Returns the boundary list
    (excluding 0 and total_duration)."""
    if total_duration <= target_chunk_sec:
        return []
    boundaries: list[float] = []
    target = target_chunk_sec
    while target < total_duration:
        best = target
        best_dist = window_sec + 1.0
        for ss, se in silences:
            mid = (ss + se) / 2.0
            dist = abs(mid - target)
            if dist < best_dist:
                best_dist = dist
                best = mid
        # Only commit a snapped boundary if it was within the window.
        if best_dist <= window_sec and best > (boundaries[-1] if boundaries else 0.0) + 1.0:
            boundaries.append(best)
        else:
            boundaries.append(target)
        target += target_chunk_sec
    return boundaries


def _base_split_cmd(src: Path, audio_cleanup: bool) -> list[str]:
    """Common ffmpeg prefix for the cleanup/split pipeline."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
    ]


def _segment_args(duration: float, silence_aware: bool, silences: list) -> list[str]:
    """Return the segment-muxer time arguments. Uses -segment_times when
    silence-aware mode has something to snap to, otherwise fixed -segment_time."""
    if silence_aware:
        boundaries = _compute_silence_boundaries(duration, silences)
        if boundaries:
            return ["-segment_times", ",".join(f"{b:.3f}" for b in boundaries)]
    return ["-segment_time", str(CHUNK_SECONDS)]


def split(
    src: Path,
    tmp: Path,
    audio_cleanup: bool = False,
    silence_aware: bool = False,
) -> list[Path]:
    log(f"Splitting {src.name}  (cleanup={audio_cleanup}, silence_aware={silence_aware})")
    out = tmp / "chunk_%03d.wav"

    silences: list[tuple[float, float]] = []
    duration = 0.0
    if silence_aware:
        duration = get_duration(src)
        silences = _find_silence_regions(src)
        log(f"  silencedetect: {len(silences)} silent region(s)")

    af_parts: list[str] = []
    if audio_cleanup:
        af_parts.append(AUDIO_CLEANUP_FILTER)

    cmd = _base_split_cmd(src, audio_cleanup) + ["-vn"]
    if af_parts:
        cmd += ["-af", ",".join(af_parts)]
    cmd += [
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment",
        *_segment_args(duration, silence_aware, silences),
        "-reset_timestamps", "1",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    chunks = sorted(tmp.glob("chunk_*.wav"))
    log(f"  {len(chunks)} chunk(s) created")
    return chunks


def _sanitize_out_name(p: Path) -> str:
    return re.sub(r"[^\w\-\. ]+", "_", p.stem) + ".txt"


def probe_audio_streams(path: Path) -> list[int]:
    """Return channel count of each audio stream in order, or [] on failure.

    A file with two stereo audio streams (common for screen-recording tools that
    capture mic + desktop separately) returns [2, 2].
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=channels",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return [int(ln.strip()) for ln in result.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _speaker_tracks_for(path: Path) -> list[dict]:
    """Decide how to split a file for per-speaker transcription.

    Returns a list of track descriptors, each:
      {"label": "Speaker N", "kind": "stream"|"channel", "index": int}

    Strategy:
    - If the file has >=2 audio streams, each stream is one speaker (downmixed
      to mono). This is the case for OBS-style recordings (mic + desktop audio).
    - Else if the single audio stream has >=2 channels, each channel is one
      speaker. This is the case for clean L/R stereo recordings.
    - Otherwise returns [] (no meaningful separation possible).
    """
    streams = probe_audio_streams(path)
    if len(streams) >= 2:
        return [
            {"label": f"Speaker {i + 1}", "kind": "stream", "index": i}
            for i in range(len(streams))
        ]
    if len(streams) == 1 and streams[0] >= 2:
        return [
            {"label": f"Speaker {i + 1}", "kind": "channel", "index": i}
            for i in range(streams[0])
        ]
    return []


def split_track(
    src: Path,
    tmp: Path,
    track: dict,
    track_idx: int,
    audio_cleanup: bool = False,
    silence_aware: bool = False,
    silences: list[tuple[float, float]] | None = None,
    duration: float = 0.0,
) -> list[Path]:
    """Extract one track (stream or channel) as mono 16k chunks.

    `silences` and `duration` should be pre-computed by the caller when
    silence_aware=True (shared across all tracks of the same file)."""
    out_pattern = tmp / f"t{track_idx}_%03d.wav"
    pre: list[str] = []
    af_parts: list[str] = []

    if track["kind"] == "stream":
        # Downmix the whole stream to mono. Handles the L==R duplicated-mono
        # case (common for OBS mic streams) correctly.
        pre = ["-map", f"0:a:{track['index']}", "-ac", "1"]
    elif track["kind"] == "channel":
        pre = ["-vn"]
        af_parts.append(f"pan=mono|c0=c{track['index']}")
    else:
        raise ValueError(f"unknown track kind: {track['kind']!r}")

    if audio_cleanup:
        af_parts.append(AUDIO_CLEANUP_FILTER)

    cmd = _base_split_cmd(src, audio_cleanup) + pre
    if af_parts:
        cmd += ["-af", ",".join(af_parts)]
    cmd += [
        "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment",
        *_segment_args(duration, silence_aware, silences or []),
        "-reset_timestamps", "1",
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(tmp.glob(f"t{track_idx}_*.wav"))


def _avoid_overwrite(out: Path) -> Path:
    """Return `out` if it doesn't exist, otherwise the same name with a
    ` (YYYY-MM-DD HH-MM-SS)` suffix before the extension. Prevents clobbering
    previous transcripts when re-running."""
    if not out.exists():
        return out
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    candidate = out.with_name(f"{out.stem} ({stamp}){out.suffix}")
    n = 2
    while candidate.exists():
        candidate = out.with_name(f"{out.stem} ({stamp}_{n}){out.suffix}")
        n += 1
    return candidate


# ---------------------------------------------------------------------------
# One-pass transcription with word-level timestamps
# ---------------------------------------------------------------------------

def _whisper_words(wav_path: Path) -> list[tuple[float, float, str]]:
    """Transcribe a chunk once with word-level timestamps.

    Returns flat list of (start, end, text) tuples. VAD is on to skip silence.
    """
    segments, _ = _model.transcribe(
        str(wav_path),
        language=_language,
        vad_filter=True,
        word_timestamps=True,
    )
    words: list[tuple[float, float, str]] = []
    for seg in segments:
        ws = getattr(seg, "words", None)
        if ws:
            for w in ws:
                try:
                    words.append((float(w.start), float(w.end), w.word))
                except (TypeError, ValueError):
                    continue
        else:
            # Fallback when word_timestamps didn't populate words (rare).
            words.append((float(seg.start), float(seg.end), seg.text))
    return words


def _assign_words_to_diar(
    words: list[tuple[float, float, str]],
    diar_segs: list[tuple[float, float, str]],
) -> list[tuple[float, float, str, str | None]]:
    """Attach a speaker label to each word by max-overlap with a diar segment.

    Diar segments are assumed sorted by start. Fallback chain for words that
    don't overlap any segment:
      1. inherit the previous word's speaker (handles brief unvoiced gaps)
      2. if we're before any diar segment, snap to the nearest segment by midpoint
         (prevents dropping opening words when pyannote's first segment starts late)
    """
    out: list[tuple[float, float, str, str | None]] = []
    last_spk: str | None = None
    for ws, we, wt in words:
        best_spk, best_ov = None, 0.0
        for ss, se, sp in diar_segs:
            if se <= ws:
                continue
            if ss >= we:
                break
            ov = min(we, se) - max(ws, ss)
            if ov > best_ov:
                best_ov = ov
                best_spk = sp

        if best_spk is not None:
            spk = best_spk
        elif last_spk is not None:
            spk = last_spk
        elif diar_segs:
            wm = (ws + we) / 2.0
            spk = min(diar_segs, key=lambda s: abs((s[0] + s[1]) / 2.0 - wm))[2]
        else:
            spk = None

        out.append((ws, we, wt, spk))
        if spk is not None:
            last_spk = spk
    return out


def _words_to_turns(
    annotated: list[tuple[float, float, str, str | None]],
    chunk_offset: float,
) -> list[tuple[str, float, str]]:
    """Group consecutive same-speaker words into turns.

    Returns list of (speaker, absolute_start_seconds, text). Words with a None
    speaker at the very start (before any diar segment) are dropped.
    """
    turns: list[tuple[str, float, str]] = []
    cur_spk: str | None = None
    cur_start: float = 0.0
    cur_parts: list[str] = []

    def flush():
        if cur_spk is None:
            return
        text = "".join(cur_parts).strip()
        if text:
            turns.append((cur_spk, cur_start + chunk_offset, text))

    for ws, _we, wt, spk in annotated:
        if spk is None:
            continue
        if spk != cur_spk:
            flush()
            cur_spk = spk
            cur_start = ws
            cur_parts = [wt]
        else:
            cur_parts.append(wt)
    flush()
    return turns


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
    silero_vad: bool = False,
) -> tuple[list[tuple[float, float, str]], dict[str, np.ndarray]]:
    """Run diarization subprocess. Returns (segments, embeddings_by_local_label).

    `speaker_hint` may contain any of: num_speakers, min_speakers, max_speakers.
    `silero_vad` asks the worker to pre-compute Silero speech regions and drop
    pyannote segments that don't meaningfully overlap them (reduces phantom
    speakers from noise).
    `embeddings_by_local_label` maps each chunk-local SPEAKER_xx to its 1-D numpy
    embedding vector, or is empty if the pipeline didn't return embeddings.
    """
    cmd = [sys.executable, str(DIARIZE_WORKER), str(wav_path)]
    if use_cpu:
        cmd.append("--cpu")
    if silero_vad:
        cmd.append("--silero-vad")

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

class GlobalLabeler:
    """Dispenses unique `Speaker N` labels. Shared across trackers so a single
    file with multiple diarized tracks keeps consecutive, non-colliding IDs."""

    def __init__(self, start: int = 1):
        self._next = start

    def next(self) -> str:
        name = f"Speaker {self._next}"
        self._next += 1
        return name


class GlobalSpeakerTracker:
    """Match chunk-local speaker labels to stable global speakers across chunks.

    Pyannote's SPEAKER_00/SPEAKER_01 labels are local to each pipeline call, so
    naive label reuse across chunks causes phantom speakers. This class keeps a
    running-mean embedding (centroid) per global speaker and matches new local
    speakers by cosine similarity.
    """

    def __init__(
        self,
        threshold: float = SPEAKER_MATCH_THRESHOLD,
        labeler: GlobalLabeler | None = None,
    ):
        self.threshold = threshold
        self._labeler = labeler or GlobalLabeler()
        self._centroids: list[np.ndarray] = []     # index = local-to-global id
        self._counts: list[int] = []
        self._names: list[str] = []

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
        name = self._labeler.next()
        self._names.append(name)
        self._counts.append(1 if vec is not None else 0)
        self._centroids.append(vec.copy() if vec is not None else np.zeros(0, dtype=np.float32))
        return name

    def assign(self, local_label: str, embedding: np.ndarray | None) -> str:
        """Return the stable global name for a local chunk speaker."""
        if embedding is None or embedding.size == 0:
            # No embedding — fall back to treating each distinct local label as new.
            # Loses cross-chunk identity but preserves within-chunk correctness.
            return self._new_speaker(None)

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

def transcribe_file(
    p: Path,
    out: Path,
    progress=_noop,
    audio_cleanup: bool = False,
    silence_aware: bool = False,
):
    log(f"File job started: {p.name}")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting audio{dur_str} ...")
        chunks = split(p, tmpdir, audio_cleanup=audio_cleanup, silence_aware=silence_aware)
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

def _emit_turns_to_lines(
    turns: list[tuple[str, float, str]],
    merged_lines: list[str],
    carry: dict,
) -> None:
    """Append turns to merged_lines, merging consecutive same-speaker turns
    across chunk boundaries. `carry` tracks the pending turn between calls:
        {"speaker": str|None, "start": float, "parts": list[str]}
    """
    for spk, start_abs, text in turns:
        if not text:
            continue
        if carry["speaker"] is None:
            carry["speaker"] = spk
            carry["start"] = start_abs
            carry["parts"] = [text]
        elif carry["speaker"] == spk:
            carry["parts"].append(text)
        else:
            joined = " ".join(carry["parts"]).strip()
            if joined:
                merged_lines.append(
                    f"[{_fmt_timestamp(carry['start'])}] {carry['speaker']}: {joined}"
                )
            carry["speaker"] = spk
            carry["start"] = start_abs
            carry["parts"] = [text]


def _flush_carry(merged_lines: list[str], carry: dict) -> None:
    if carry["speaker"] is None:
        return
    joined = " ".join(carry["parts"]).strip()
    if joined:
        merged_lines.append(
            f"[{_fmt_timestamp(carry['start'])}] {carry['speaker']}: {joined}"
        )
    carry["speaker"] = None
    carry["start"] = 0.0
    carry["parts"] = []


def transcribe_file_diarized(
    p: Path,
    out: Path,
    progress=_noop,
    speaker_hint: dict | None = None,
    audio_cleanup: bool = False,
    silence_aware: bool = False,
    silero_vad: bool = False,
):
    log(f"Diarized file job started: {p.name}")
    if speaker_hint:
        log(f"Speaker hint: {speaker_hint}")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_diar_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting audio{dur_str} ...")
        chunks = split(p, tmpdir, audio_cleanup=audio_cleanup, silence_aware=silence_aware)
        total = len(chunks)
        w = len(str(total))
        t0 = time.time()

        tracker = GlobalSpeakerTracker()
        merged_lines: list[str] = []
        carry: dict = {"speaker": None, "start": 0.0, "parts": []}
        chunk_offset = 0.0

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

            diar_segs, emb_by_label = _diarize_wav(
                chunk, speaker_hint=speaker_hint, silero_vad=silero_vad
            )
            active = [s for s in diar_segs if s[1] - s[0] >= 0.15]

            # Resolve chunk-local labels to stable global names via embeddings.
            local_to_global: dict[str, str] = {}
            for _s, _e, spk in active:
                if spk not in local_to_global:
                    local_to_global[spk] = tracker.assign(spk, emb_by_label.get(spk))

            # Rewrite segments with global speaker names so word assignment
            # operates in the stable label space.
            global_segs = [
                (s, e, local_to_global[spk]) for (s, e, spk) in active
            ]
            global_segs.sort(key=lambda x: x[0])

            progress(
                f"Chunk {idx:{w}}/{total}  —  transcribing whole chunk ..."
            )

            # One Whisper call per chunk instead of one per diar segment.
            words = _whisper_words(chunk)
            annotated = _assign_words_to_diar(words, global_segs)
            turns = _words_to_turns(annotated, chunk_offset)
            _emit_turns_to_lines(turns, merged_lines, carry)

            # Advance chunk offset using the actual decoded chunk duration.
            chunk_offset += get_duration(chunk)

            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx)
            progress(
                f"Chunk {idx:{w}}/{total}  done"
                f"  |  elapsed {fmt_elapsed(elapsed)}"
                f"  |  ETA {fmt_eta(eta)}"
            )

        _flush_carry(merged_lines, carry)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(merged_lines).strip() + "\n", encoding="utf-8")
        log(
            f"Diarized job finished in {fmt_elapsed(time.time() - t0)} "
            f"-> {out.name}  ({len(tracker._names)} speaker(s))"
        )

    finally:
        _cleanup_tmp(tmpdir)


# ---------------------------------------------------------------------------
# Per-channel transcription (for stereo recordings where each channel is one speaker)
# ---------------------------------------------------------------------------

def transcribe_file_multi_track(
    p: Path,
    out: Path,
    progress=_noop,
    track_speakers: list | None = None,
    audio_cleanup: bool = False,
    silence_aware: bool = False,
    silero_vad: bool = False,
):
    """Transcribe a file with independent per-speaker audio tracks.

    Layouts handled:
    - Multiple audio streams (e.g. OBS mic + Discord capture): each stream
      downmixed to mono. Avoids the 'L==R duplicated mono' trap.
    - Single stream with multiple channels: each channel is one track.

    `track_speakers` is a list aligned with tracks; each element may be:
      - an int (1 = single speaker, no diarization; N>=2 = diarize with
        max_speakers=N within that track)
      - None (auto-diarize within that track; pyannote picks the count)
    Missing / extra entries default to 1 (single speaker).
    """
    tracks = _speaker_tracks_for(p)
    if not tracks:
        raise RuntimeError(
            f"no suitable tracks for speaker separation in {p.name} "
            f"(need >=2 audio streams, or >=2 channels in one stream)"
        )

    # Align track_speakers with the detected tracks.
    hints: list[int | None] = list(track_speakers or [])
    while len(hints) < len(tracks):
        hints.append(1)
    hints = hints[:len(tracks)]

    kind_summary = (
        f"{len(tracks)} streams"
        if tracks[0]["kind"] == "stream"
        else f"{len(tracks)} channels of stream 0"
    )
    hint_summary = ", ".join(
        "auto" if h is None else ("solo" if h == 1 else f"<={h}")
        for h in hints
    )
    log(f"Multi-track job started: {p.name}  ({kind_summary}; speakers: {hint_summary})")
    duration = get_duration(p)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_mt_"))

    try:
        dur_str = f"  ({fmt_eta(duration)} source)" if duration else ""
        progress(f"Splitting {kind_summary}{dur_str} ...")

        # Silence detection is done once on the source (raw) and reused for
        # every track so all tracks share the same chunk boundaries in source time.
        silences: list[tuple[float, float]] = []
        if silence_aware:
            silences = _find_silence_regions(p)
            log(f"  silencedetect: {len(silences)} silent region(s)")

        track_chunks: list[list[Path]] = []
        for t_idx, track in enumerate(tracks):
            track_chunks.append(split_track(
                p, tmpdir, track, t_idx,
                audio_cleanup=audio_cleanup,
                silence_aware=silence_aware,
                silences=silences,
                duration=duration,
            ))

        total = max(len(c) for c in track_chunks)
        if total == 0:
            raise RuntimeError("ffmpeg produced no chunks")
        w = len(str(total))
        t0 = time.time()

        # Shared labeler: every Speaker N across every track comes from here,
        # so numbering stays unique and (roughly) in order of first speech.
        labeler = GlobalLabeler()

        # Per-track state:
        #   "solo_label": lazily-allocated name for a 1-speaker track
        #   "tracker":    GlobalSpeakerTracker for diarized tracks
        track_state: list[dict] = []
        for i, hint in enumerate(hints):
            if hint == 1:
                track_state.append({"kind": "solo", "label": None})
            else:
                # None (auto) or int>=2 → diarize within this track.
                sub_hint: dict | None = {"max_speakers": hint} if isinstance(hint, int) and hint >= 2 else None
                track_state.append({
                    "kind": "diar",
                    "tracker": GlobalSpeakerTracker(labeler=labeler),
                    "sub_hint": sub_hint,
                })

        merged_lines: list[str] = []
        carry: dict = {"speaker": None, "start": 0.0, "parts": []}
        chunk_offset = 0.0

        for idx in range(total):
            elapsed_pre = time.time() - t0
            eta_str = (
                f"  |  ETA ~{fmt_eta((elapsed_pre / idx) * (total - idx))}"
                if idx > 0 else ""
            )
            progress(
                f"Chunk {idx + 1:{w}}/{total}  —  processing {len(tracks)} track(s) ..."
                f"  |  elapsed {fmt_elapsed(elapsed_pre)}{eta_str}"
            )

            all_words: list[tuple[float, float, str, str | None]] = []
            chunk_durations: list[float] = []

            for t_idx, track in enumerate(tracks):
                if idx >= len(track_chunks[t_idx]):
                    continue
                chunk_path = track_chunks[t_idx][idx]
                chunk_durations.append(get_duration(chunk_path))

                state = track_state[t_idx]
                words = _whisper_words(chunk_path)
                if not words:
                    continue

                if state["kind"] == "solo":
                    if state["label"] is None:
                        state["label"] = labeler.next()
                    label = state["label"]
                    for ws, we, wt in words:
                        all_words.append((ws, we, wt, label))
                else:
                    # Diarize this track's chunk and assign its words.
                    diar_segs, emb_by_label = _diarize_wav(
                        chunk_path,
                        speaker_hint=state["sub_hint"],
                        silero_vad=silero_vad,
                    )
                    active = [s for s in diar_segs if s[1] - s[0] >= 0.15]

                    tracker: GlobalSpeakerTracker = state["tracker"]
                    local_to_global: dict[str, str] = {}
                    for _s, _e, spk in active:
                        if spk not in local_to_global:
                            local_to_global[spk] = tracker.assign(spk, emb_by_label.get(spk))

                    global_segs = [
                        (s, e, local_to_global[spk]) for (s, e, spk) in active
                    ]
                    global_segs.sort(key=lambda x: x[0])

                    annotated = _assign_words_to_diar(words, global_segs)
                    for ws, we, wt, spk in annotated:
                        all_words.append((ws, we, wt, spk))

            all_words.sort(key=lambda x: x[0])
            turns = _words_to_turns(all_words, chunk_offset)
            _emit_turns_to_lines(turns, merged_lines, carry)

            chunk_offset += max(chunk_durations) if chunk_durations else 0.0

            elapsed = time.time() - t0
            eta = (elapsed / (idx + 1)) * (total - idx - 1)
            progress(
                f"Chunk {idx + 1:{w}}/{total}  done"
                f"  |  elapsed {fmt_elapsed(elapsed)}"
                f"  |  ETA {fmt_eta(eta)}"
            )

        _flush_carry(merged_lines, carry)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(merged_lines).strip() + "\n", encoding="utf-8")
        log(f"Multi-track job finished in {fmt_elapsed(time.time() - t0)} -> {out.name}")

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


def _parse_track_speakers(raw) -> list | None:
    """Parse a per-track speaker-count list from the payload.

    Accepts a list whose elements are int or None (JSON null). Non-positive
    ints become None (auto). Non-list input returns None.
    """
    if not isinstance(raw, list):
        return None
    out: list = []
    for v in raw:
        if v is None:
            out.append(None)
        else:
            try:
                n = int(v)
            except (TypeError, ValueError):
                out.append(None)
                continue
            out.append(n if n >= 1 else None)
    return out or None


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
            out = _avoid_overwrite(p.with_name(_sanitize_out_name(p)))
            hint = _extract_speaker_hint(d)
            multi_track = bool(d.get("multi_track") or d.get("per_channel"))
            track_speakers = _parse_track_speakers(d.get("track_speakers"))
            audio_cleanup = bool(d.get("audio_cleanup"))
            silence_aware = bool(d.get("silence_aware"))
            silero_vad = bool(d.get("silero_vad"))

            with job_lock:
                if multi_track:
                    transcribe_file_multi_track(
                        p, out, progress=progress,
                        track_speakers=track_speakers,
                        audio_cleanup=audio_cleanup,
                        silence_aware=silence_aware,
                        silero_vad=silero_vad,
                    )
                elif op == "TRANSCRIBE_FILE_DIARIZED" or d.get("diarize") is True:
                    transcribe_file_diarized(
                        p, out, progress=progress, speaker_hint=hint,
                        audio_cleanup=audio_cleanup,
                        silence_aware=silence_aware,
                        silero_vad=silero_vad,
                    )
                else:
                    transcribe_file(
                        p, out, progress=progress,
                        audio_cleanup=audio_cleanup,
                        silence_aware=silence_aware,
                    )

            resp = f"OK {out}"

        elif op in ("TRANSCRIBE_FOLDER", "TRANSCRIBE_FOLDER_DIARIZED"):
            d = json.loads(payload)
            root = Path(d["path"])
            inc = d.get("include_subfolders", False)
            mir = d.get("mirror_structure", False)
            diarize = (op == "TRANSCRIBE_FOLDER_DIARIZED") or (d.get("diarize") is True)
            hint = _extract_speaker_hint(d) if diarize else None
            multi_track_pref = bool(d.get("multi_track") or d.get("per_channel"))
            track_speakers = _parse_track_speakers(d.get("track_speakers"))
            audio_cleanup = bool(d.get("audio_cleanup"))
            silence_aware = bool(d.get("silence_aware"))
            silero_vad = bool(d.get("silero_vad"))

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
                    out = _avoid_overwrite(out)

                    # Multi-track is a preference: apply when the file actually
                    # has a usable layout, otherwise fall back to diarization.
                    use_multi_track = False
                    if multi_track_pref and diarize:
                        if _speaker_tracks_for(f):
                            use_multi_track = True
                        else:
                            progress(
                                f"  {f.name} has no multi-track layout — "
                                f"using normal diarization"
                            )

                    if use_multi_track:
                        transcribe_file_multi_track(
                            f, out, progress=progress,
                            track_speakers=track_speakers,
                            audio_cleanup=audio_cleanup,
                            silence_aware=silence_aware,
                            silero_vad=silero_vad,
                        )
                    elif diarize:
                        transcribe_file_diarized(
                            f, out, progress=progress, speaker_hint=hint,
                            audio_cleanup=audio_cleanup,
                            silence_aware=silence_aware,
                            silero_vad=silero_vad,
                        )
                    else:
                        transcribe_file(
                            f, out, progress=progress,
                            audio_cleanup=audio_cleanup,
                            silence_aware=silence_aware,
                        )

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
