import socket
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765


def daemon_available():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect((HOST, PORT))
            s.sendall(b"PING\n")
            return s.recv(1024).decode().startswith("OK")
    except Exception:
        return False


def daemon_send_streaming(line: str, timeout: int = 86400) -> str:
    """
    Send a command and stream PROGRESS lines to stdout as they arrive.
    Returns the final OK/ERR response line.

    The daemon sends zero or more lines prefixed with "PROGRESS " followed
    by a single final line starting with "OK" or "ERR".
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((HOST, PORT))
        s.sendall((line + "\n").encode("utf-8"))

        buf = ""
        final = None

        while final is None:
            try:
                chunk = s.recv(4096).decode("utf-8", errors="replace")
            except socket.timeout:
                break

            if not chunk:
                break

            buf += chunk

            while "\n" in buf:
                msg, buf = buf.split("\n", 1)
                msg = msg.strip()
                if not msg:
                    continue

                if msg.startswith("PROGRESS "):
                    # Strip the prefix and print with a timestamp so the
                    # console window shows live updates during long jobs.
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] {msg[9:]}", flush=True)

                elif msg.startswith("OK") or msg.startswith("ERR"):
                    final = msg
                    break   # ignore anything left in buf after the final line

        return final or buf.strip() or "ERR no response"


def yn(q, default=False):
    suf = " [Y/n] " if default else " [y/N] "
    while True:
        a = input(q + suf).strip().lower()
        if not a:
            return default
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


def probe_audio_streams(path: Path) -> list[int]:
    """Return channel count per audio stream, or [] on failure or no ffprobe."""
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


def ask_track_speakers(track_count: int, track_kind: str) -> list:
    """Prompt per track for expected speaker count.

    Each entry in the returned list is:
      - int >= 1 (1 = solo, no diarization; N>=2 = diarize with max N)
      - None (auto-detect within this track)
    Default when user hits Enter is 1 (solo) — matches the common OBS case of
    'mic = just me, Discord = multiple'.
    """
    label = "Stream" if track_kind == "stream" else "Channel"
    print(
        f"\nFor each {label.lower()}, enter expected speaker count:\n"
        f"  Enter      = 1 (solo, no diarization)\n"
        f"  N (>=2)    = diarize, up to N speakers\n"
        f"  'auto'     = diarize, let pyannote pick\n"
    )
    result: list = []
    for i in range(track_count):
        while True:
            raw = input(f"{label} {i + 1} speakers [1]: ").strip().lower()
            if not raw:
                result.append(1)
                break
            if raw in ("auto", "a"):
                result.append(None)
                break
            try:
                n = int(raw)
                if n < 1:
                    raise ValueError
                result.append(n)
                break
            except ValueError:
                print("  Enter a positive integer, 'auto', or blank.")
    return result


def ask_folder_track_speakers() -> list:
    """Ask up to 2 default track-speaker counts for folder jobs.

    Folder files can have heterogeneous layouts; we accept defaults for the
    first two tracks (covers the common OBS mic + Discord case). Extra tracks
    default to 1 (solo). User can still choose auto per slot.
    """
    print(
        "\nFolder multi-track defaults — applied to each file's tracks in order.\n"
        "Leave blank to default to 1 (solo).\n"
    )
    result: list = []
    for i in range(2):
        while True:
            raw = input(
                f"Track {i + 1} default speakers [1] "
                f"(integer, 'auto', or blank): "
            ).strip().lower()
            if not raw:
                result.append(1)
                break
            if raw in ("auto", "a"):
                result.append(None)
                break
            try:
                n = int(raw)
                if n < 1:
                    raise ValueError
                result.append(n)
                break
            except ValueError:
                print("  Enter a positive integer, 'auto', or blank.")
    return result


def ask_speaker_hint() -> dict:
    """Prompt for speaker count. Returns dict suitable for the socket payload.

    Accepts:
        blank   -> auto-detect (empty dict)
        N       -> max_speakers=N (cap; pyannote may find fewer)
        N-M     -> min_speakers=N, max_speakers=M
    """
    while True:
        raw = input(
            "Max speakers in recording?  (blank = auto, e.g. '3' or '2-4'): "
        ).strip()
        if not raw:
            return {}
        if "-" in raw:
            try:
                lo, hi = raw.split("-", 1)
                lo_i, hi_i = int(lo.strip()), int(hi.strip())
                if lo_i <= 0 or hi_i <= 0 or lo_i > hi_i:
                    raise ValueError
                return {"min_speakers": lo_i, "max_speakers": hi_i}
            except ValueError:
                print("  Expected a range like '2-4'. Try again.")
                continue
        try:
            n = int(raw)
            if n <= 0:
                raise ValueError
            return {"max_speakers": n}
        except ValueError:
            print("  Expected a positive integer or range. Try again.")
            continue


def main(argv):
    if len(argv) < 2:
        print("Usage: transcribe_drop.py <file_or_folder> ...")
        return 2

    use_daemon = daemon_available()

    for a in argv[1:]:
        p = Path(a)
        if not p.exists():
            print(f"Path not found, skipping: {a}")
            continue

        if p.is_file():
            audio_cleanup = yn(
                "Apply audio cleanup? (highpass + gentle compressor + loudness norm)",
                False,
            )
            silence_aware = yn(
                "Use silence-aware chunk boundaries? (snap 15-min cuts to silences)",
                False,
            )

            diar = yn("Add speaker labels (diarization)?", False)
            silero_vad = False
            if diar:
                silero_vad = yn(
                    "Use Silero VAD to drop noise-triggered phantom speakers?",
                    False,
                )
            hint: dict = {}
            multi_track = False
            track_speakers: list = []
            track_kind = ""
            if diar:
                streams = probe_audio_streams(p)
                if len(streams) >= 2:
                    multi_track = yn(
                        f"File has {len(streams)} audio streams "
                        f"(likely mic + system audio). "
                        f"Treat each stream as a separate speaker?",
                        True,
                    )
                    if multi_track:
                        track_kind = "stream"
                        track_speakers = ask_track_speakers(len(streams), "stream")
                elif len(streams) == 1 and streams[0] >= 2:
                    multi_track = yn(
                        f"File has {streams[0]} audio channels in one stream. "
                        f"Treat each channel as a separate speaker? "
                        f"(only say yes if you know channels are cleanly split)",
                        False,
                    )
                    if multi_track:
                        track_kind = "channel"
                        track_speakers = ask_track_speakers(streams[0], "channel")
                if not multi_track:
                    hint = ask_speaker_hint()
            if use_daemon:
                op = "TRANSCRIBE_FILE_DIARIZED" if diar else "TRANSCRIBE_FILE"
                payload = {"path": str(p), **hint}
                if multi_track:
                    payload["multi_track"] = True
                    if track_speakers:
                        payload["track_speakers"] = track_speakers
                if audio_cleanup:
                    payload["audio_cleanup"] = True
                if silence_aware:
                    payload["silence_aware"] = True
                if silero_vad:
                    payload["silero_vad"] = True
                print(f"\nStarting job — progress will appear below.\n")
                result = daemon_send_streaming(op + " " + json.dumps(payload))
                print(f"\n{result}")
            else:
                print("Daemon not running. Start it for GPU batch.")

        else:
            inc = yn("Include subfolders?", False)
            mir = False
            if inc:
                mir = yn("Mirror subfolder structure?", True)

            audio_cleanup = yn(
                "Apply audio cleanup to every file? (highpass + compressor + loudnorm)",
                False,
            )
            silence_aware = yn(
                "Use silence-aware chunk boundaries?",
                False,
            )

            diar = yn("Add speaker labels (diarization)?", False)
            silero_vad = False
            if diar:
                silero_vad = yn(
                    "Use Silero VAD to drop noise-triggered phantom speakers?",
                    False,
                )
            hint = {}
            multi_track = False
            track_speakers = []
            if diar:
                multi_track = yn(
                    "Separate speakers by audio stream/channel where possible? "
                    "(mono files fall back to diarization)",
                    False,
                )
                if multi_track:
                    track_speakers = ask_folder_track_speakers()
                else:
                    hint = ask_speaker_hint()

            if use_daemon:
                op = "TRANSCRIBE_FOLDER_DIARIZED" if diar else "TRANSCRIBE_FOLDER"
                payload_dict = {
                    "path": str(p),
                    "include_subfolders": inc,
                    "mirror_structure": mir,
                    **hint,
                }
                if multi_track:
                    payload_dict["multi_track"] = True
                    if track_speakers:
                        payload_dict["track_speakers"] = track_speakers
                if audio_cleanup:
                    payload_dict["audio_cleanup"] = True
                if silence_aware:
                    payload_dict["silence_aware"] = True
                if silero_vad:
                    payload_dict["silero_vad"] = True
                payload = json.dumps(payload_dict)
                print(f"\nStarting folder job — progress will appear below.\n")
                result = daemon_send_streaming(op + " " + payload)
                print(f"\n{result}")
            else:
                print("Daemon not running. Start it for GPU batch.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
