# ======================================
# FILE: dexsdk/net/publisher.py
# ======================================
"""Simple UDP JSON publisher for vision results."""

import json
import socket
from typing import Any, Tuple


class UDPPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 40001):
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr: Tuple[str, int] = (self.host, self.port)

    def configure(self, host: str | None = None, port: int | None = None):
        """Update IP/port at runtime."""
        if host is not None:
            self.host = host
        if port is not None:
            self.port = int(port)
        self._addr = (self.host, self.port)

    def send(self, payload: Any) -> bool:
        """Serialize payload as compact JSON and send via UDP.
        Returns True if best-effort send attempted without local exception.
        """
        try:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            # NOTE: Typical safe UDP payload size < ~64KB; our payloads are tiny.
            self._sock.sendto(data, self._addr)
            return True
        except Exception:
            # Best-effort: drop on error to keep UI real-time.
            return False

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass
