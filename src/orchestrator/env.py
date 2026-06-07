"""Minimal .env loader for orchestrator configuration."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    """Load KEY=VALUE lines into os.environ if not already set."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def load_project_env(project_root: Path) -> dict[str, str]:
    """Load root .env first, then orchestrator/.env for local overrides."""
    values: dict[str, str] = {}
    for path in [project_root / ".env", project_root / "orchestrator" / ".env"]:
        values.update(load_dotenv(path))
    return values
