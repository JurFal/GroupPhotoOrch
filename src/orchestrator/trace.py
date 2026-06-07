"""Trace persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import AgentTrace


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trace(path: Path, trace: AgentTrace) -> None:
    write_json(path, trace.to_dict())
