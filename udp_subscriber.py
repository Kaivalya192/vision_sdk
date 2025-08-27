# =========================
# FILE: udp_subscriber.py
# =========================
import socket
import json

def main(host="0.0.0.0", port=40001):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"📡 Listening for UDP JSON packets on {host}:{port} ...")

    while True:
        data, addr = sock.recvfrom(65535)
        try:
            msg = json.loads(data.decode("utf-8"))
            print(f"\n--- Message from {addr} ---")
            print(json.dumps(msg, indent=2))
        except Exception as e:
            print(f"[Error decoding JSON from {addr}] {e}")

if __name__ == "__main__":
    main()
