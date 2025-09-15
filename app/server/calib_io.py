from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_k_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p


def load_k_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

