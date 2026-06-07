"""Tool registry loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    mode: str
    raw: dict[str, Any]


def load_registry(path: Path) -> dict[str, ToolSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = {}
    for name, raw in data.get("tools", {}).items():
        tools[name] = ToolSpec(
            name=name,
            description=raw.get("description", ""),
            mode=raw.get("mode", "unknown"),
            raw=raw,
        )
    return tools
