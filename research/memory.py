"""Long-term research memory. The file is gitignored and stays on this machine."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "research_memory.json"
_lock = threading.Lock()


def _read() -> dict:
    if not PATH.exists():
        return {}
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load(symbol: str) -> str:
    row = _read().get((symbol or "").upper()) or {}
    memo = row.get("memo") if isinstance(row, dict) else ""
    return memo or ""


def save(symbol: str, memo: str, run_id: str) -> None:
    key = (symbol or "").upper()
    if not key:
        return
    with _lock:
        data = _read()
        data[key] = {
            "memo": memo,
            "runId": run_id,
            "asOf": datetime.now(timezone.utc).isoformat(),
        }
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
