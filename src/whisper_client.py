import socket
import sys

HOST = "127.0.0.1"
PORT = 8765

def main():
    args = sys.argv[1:]

    # Parse optional --output <path> flag
    output_path = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_path = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if not args:
        resp = "ERR missing_command"
        print(resp)
        if output_path:
            _write(output_path, resp)
        sys.exit(2)

    cmd = " ".join(args).strip()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30.0)
            s.connect((HOST, PORT))
            s.sendall((cmd + "\n").encode("utf-8"))
            resp = s.recv(1024 * 1024).decode("utf-8", errors="replace").strip()
    except Exception as e:
        resp = f"ERR {e}"

    print(resp)
    if output_path:
        _write(output_path, resp)


def _write(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"ERR failed to write output file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()