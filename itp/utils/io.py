from __future__ import annotations

from pathlib import Path

def read_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

