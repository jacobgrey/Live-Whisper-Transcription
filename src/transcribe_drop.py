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


def get_channel_count(path: Path) -> int:
    """Return channel count of first audio stream, 0 if undetectable or ffprobe missing."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=channels",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return int(result.stdout.strip() or "0")
    except Exception:
        return 0


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
            diar = yn("Add speaker labels (diarization)?", False)
            hint: dict = {}
            per_channel = False
            if diar:
                ch = get_channel_count(p)
                if ch >= 2:
                    per_channel = yn(
                        f"File has {ch} audio channel(s). Treat each channel as a separate speaker?",
                        False,
                    )
                if not per_channel:
                    hint = ask_speaker_hint()
            if use_daemon:
                op = "TRANSCRIBE_FILE_DIARIZED" if diar else "TRANSCRIBE_FILE"
                payload = {"path": str(p), **hint}
                if per_channel:
                    payload["per_channel"] = True
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

            diar = yn("Add speaker labels (diarization)?", False)
            hint = {}
            per_channel = False
            if diar:
                per_channel = yn(
                    "Use per-channel mode for stereo files? (mono files fall back to normal diarization)",
                    False,
                )
                if not per_channel:
                    hint = ask_speaker_hint()

            if use_daemon:
                op = "TRANSCRIBE_FOLDER_DIARIZED" if diar else "TRANSCRIBE_FOLDER"
                payload_dict = {
                    "path": str(p),
                    "include_subfolders": inc,
                    "mirror_structure": mir,
                    **hint,
                }
                if per_channel:
                    payload_dict["per_channel"] = True
                payload = json.dumps(payload_dict)
                print(f"\nStarting folder job — progress will appear below.\n")
                result = daemon_send_streaming(op + " " + payload)
                print(f"\n{result}")
            else:
                print("Daemon not running. Start it for GPU batch.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
