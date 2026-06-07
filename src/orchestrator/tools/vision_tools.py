"""Vision tool wrappers used by the orchestrator."""
from __future__ import annotations

from pathlib import Path

from ..adapters.sam3_artifact_adapter import inspect_existing_artifacts as _inspect
from ..schemas import CaseConfig, Observation


def inspect_existing_artifacts(
    case: CaseConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
    **_: object,
) -> Observation:
    del output_dir, dry_run
    return _inspect(case)


__all__ = ["inspect_existing_artifacts"]
