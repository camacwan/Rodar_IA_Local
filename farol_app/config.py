"""Configuração local em ~/.farol (token, benchmark, preferências)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

HOME = Path(os.environ.get("FAROL_HOME", Path.home() / ".farol"))
FILE = HOME / "config.json"
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(FILE.read_text())
    except Exception:
        return {}


def _save(data: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try:
        os.chmod(tmp, 0o600)  # o token fica legível só pelo seu usuário
    except OSError:
        pass
    tmp.replace(FILE)


def get(key: str, default=None):
    return _load().get(key, default)


def set(key: str, value) -> None:  # noqa: A001
    with _lock:
        d = _load()
        d[key] = value
        _save(d)


def delete(key: str) -> None:
    with _lock:
        d = _load()
        d.pop(key, None)
        _save(d)


def models_dir() -> Path:
    p = Path(get("models_dir") or HOME / "models")
    p.mkdir(parents=True, exist_ok=True)
    return p
