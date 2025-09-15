from __future__ import annotations

"""Simple JSON cache helpers."""

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_DIR = Path.home() / ".vim_rb"


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = DEFAULT_DIR / p
    return p


def save_json(path: str, data: Any) -> Path:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def load_json(path: str) -> Dict[str, Any] | None:
    p = _resolve(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

