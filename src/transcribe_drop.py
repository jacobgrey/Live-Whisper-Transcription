import socket
import json
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
            if use_daemon:
                op = "TRANSCRIBE_FILE_DIARIZED" if diar else "TRANSCRIBE_FILE"
                print(f"\nStarting job — progress will appear below.\n")
                result = daemon_send_streaming(op + " " + json.dumps({"path": str(p)}))
                print(f"\n{result}")
            else:
                print("Daemon not running. Start it for GPU batch.")

        else:
            inc = yn("Include subfolders?", False)
            mir = False
            if inc:
                mir = yn("Mirror subfolder structure?", True)

            diar = yn("Add speaker labels (diarization)?", False)

            if use_daemon:
                op = "TRANSCRIBE_FOLDER_DIARIZED" if diar else "TRANSCRIBE_FOLDER"
                payload = json.dumps(
                    {
                        "path": str(p),
                        "include_subfolders": inc,
                        "mirror_structure": mir,
                    }
                )
                print(f"\nStarting folder job — progress will appear below.\n")
                result = daemon_send_streaming(op + " " + payload)
                print(f"\n{result}")
            else:
                print("Daemon not running. Start it for GPU batch.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
