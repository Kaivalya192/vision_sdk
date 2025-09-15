from __future__ import annotations

import socket
import time
from typing import Literal


Mode = Literal["training", "trigger"]


class LEDClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 12345):
        self.host = host
        self.port = port
        self._state = False

    def send(self, on: bool):
        if on == self._state:
            return
        try:
            with socket.create_connection((self.host, self.port), timeout=0.2) as s:
                s.sendall(b"1" if on else b"0")
                try:
                    s.recv(32)
                except Exception:
                    pass
            self._state = on
        except Exception:
            # assume success to avoid hammering
            self._state = on


class LEDStateMachine:
    def __init__(self, client: LEDClient, pre_ms: int = 200, post_ms: int = 100):
        self.client = client
        self.pre_ms = int(pre_ms)
        self.post_ms = int(post_ms)
        self._training_on = False
        self._phase = "idle"  # idle|pre|capture|post
        self._t0 = 0.0

    def set_training(self, training: bool):
        if training and not self._training_on:
            self.client.send(True)
            self._training_on = True
        elif not training and self._training_on:
            self.client.send(False)
            self._training_on = False

    def trigger(self):
        if self._phase == "idle":
            self.client.send(True)
            self._phase = "pre"
            self._t0 = time.time()

    def update(self, mode: Mode) -> bool:
        """Advance the state machine; return True if capture should occur now."""
        now = time.time()

        if mode == "training":
            self.set_training(True)
            self._phase = "idle"
            return False

        # trigger mode
        self.set_training(False)

        if self._phase == "pre" and (now - self._t0) * 1000 >= self.pre_ms:
            self._phase = "capture"
            return True

        if self._phase == "post" and (now - self._t0) * 1000 >= self.post_ms:
            self.client.send(False)
            self._phase = "idle"

        return False

    def notify_captured(self):
        # move to post phase
        self._phase = "post"
        self._t0 = time.time()

