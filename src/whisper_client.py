import socket
import sys

HOST = "127.0.0.1"
PORT = 8765

def main():
    if len(sys.argv) < 2:
        print("ERR missing_command")
        sys.exit(2)

    # Allow: COMMAND [payload...]
    cmd = " ".join(sys.argv[1:]).strip()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30.0)
        s.connect((HOST, PORT))
        s.sendall((cmd + "\n").encode("utf-8"))
        resp = s.recv(1024 * 1024).decode("utf-8", errors="replace").strip()
        print(resp)

if __name__ == "__main__":
    main()