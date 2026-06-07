"""Subprocess runner used by future command-line tools."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .schemas import Observation


def run_command(command: list[str], cwd: Path, timeout_s: int = 600, dry_run: bool = False) -> Observation:
    if dry_run:
        return Observation(
            status="skipped",
            summary="Dry run: command was not executed.",
            data={"command": command, "cwd": str(cwd)},
        )
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        return Observation(status="failed", summary=f"Command failed to start: {exc!r}")

    status = "success" if result.returncode == 0 else "failed"
    return Observation(
        status=status,
        summary=f"Command exited with code {result.returncode}.",
        stdout=result.stdout,
        stderr=result.stderr,
        data={"command": command, "cwd": str(cwd), "return_code": result.returncode},
    )
