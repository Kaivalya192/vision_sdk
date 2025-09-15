from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class RobotMap:
    def __init__(self):
        self.pairs: List[Tuple[float, float, float, float]] = []  # (rx, ry, px, py)
        self.params: Optional[dict] = None

    def add_pair(self, robot_xy: List[float], pixel_xy: List[float]) -> None:
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        px, py = float(pixel_xy[0]), float(pixel_xy[1])
        self.pairs.append((rx, ry, px, py))

    def clear(self) -> None:
        self.pairs.clear()
        self.params = None

    def count(self) -> int:
        return len(self.pairs)

    def solve(self) -> dict:
        if len(self.pairs) < 2:
            raise RuntimeError("Need at least 2 pairs to solve similarity")
        R = np.array([[p[0], p[1]] for p in self.pairs], dtype=np.float64)  # robot
        P = np.array([[p[2], p[3]] for p in self.pairs], dtype=np.float64)  # pixel
        r_mean = R.mean(axis=0)
        p_mean = P.mean(axis=0)
        Rc = R - r_mean
        Pc = P - p_mean
        # Compute similarity mapping P -> R: minimize || s*U*Pc + t - Rc ||
        # Using Umeyama-like solution
        cov = Pc.T @ Rc / len(P)
        U, S, Vt = np.linalg.svd(cov)
        Rmat = U @ Vt
        if np.linalg.det(Rmat) < 0:
            Vt[-1, :] *= -1
            Rmat = U @ Vt
        var_P = (Pc ** 2).sum() / len(P)
        s = float(np.trace(np.diag(S)) / max(1e-12, var_P))
        t = r_mean - s * (Rmat @ p_mean)
        # RMSE
        pred = (s * (P @ Rmat.T)) + t
        rmse = float(np.sqrt(np.mean(np.sum((pred - R) ** 2, axis=1))))
        theta = math.degrees(math.atan2(Rmat[1, 0], Rmat[0, 0]))
        self.params = {"scale": s, "theta_deg": theta, "tx": float(t[0]), "ty": float(t[1]), "rmse": rmse}
        return dict(self.params)

    def apply(self, px: float, py: float) -> Tuple[float, float]:
        if not self.params:
            raise RuntimeError("solve() not run")
        s = self.params["scale"]
        th = math.radians(self.params["theta_deg"])
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float64)
        tx, ty = self.params["tx"], self.params["ty"]
        v = np.array([px, py], dtype=np.float64)
        out = s * (Rm @ v) + np.array([tx, ty])
        return float(out[0]), float(out[1])

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **(self.params or {}),
            "pairs": [[rx, ry, px, py] for (rx, ry, px, py) in self.pairs],
            "created_at_ms": int(time.time() * 1000),
        }
        p.write_text(json.dumps(payload, indent=2))
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            self.params = {k: data[k] for k in ("scale", "theta_deg", "tx", "ty", "rmse") if k in data}
            self.pairs = [tuple(map(float, q)) for q in data.get("pairs", [])]
        except Exception:
            self.params = None
            self.pairs = []

