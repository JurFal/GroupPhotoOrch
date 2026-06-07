"""Inspect current Computer-Graphics SAM3/person_metadata artifacts without schema normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import CaseConfig, Observation


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_existing_artifacts(case: CaseConfig) -> Observation:
    paths = case.paths()
    artifacts: list[str] = []
    warnings: list[str] = []
    data: dict[str, Any] = {"paths": paths.to_strings()}

    for label, path in [
        ("group_image", paths.group_image),
        ("person_image", paths.person_image),
        ("group_meta", paths.group_meta),
        ("person_meta", paths.person_meta),
        ("group_sam3_dir", paths.group_sam3_dir),
        ("person_sam3_dir", paths.person_sam3_dir),
    ]:
        if path.exists():
            artifacts.append(str(path))
        else:
            warnings.append(f"missing {label}: {path}")

    for label, meta_path in [("group", paths.group_meta), ("person", paths.person_meta)]:
        if not meta_path.exists():
            continue
        try:
            meta = _read_json(meta_path)
        except Exception as exc:
            warnings.append(f"cannot read {label} metadata: {exc!r}")
            continue
        persons = meta.get("persons", [])
        data[f"{label}_person_count"] = len(persons)
        data[f"{label}_warnings"] = meta.get("warnings", [])
        data[f"{label}_source_metadata"] = meta.get("source_metadata")
        if meta.get("source_metadata"):
            source_path = Path(meta["source_metadata"])
            if not source_path.is_absolute():
                source_path = meta_path.parent.parent / source_path
            if source_path.exists():
                artifacts.append(str(source_path))
            else:
                warnings.append(f"{label} source_metadata path does not exist: {source_path}")

    for label, sam3_dir in [("group", paths.group_sam3_dir), ("person", paths.person_sam3_dir)]:
        metadata_path = sam3_dir / f"{case.group_id if label == 'group' else case.person_id}.json"
        instances_dir = sam3_dir / "instances"
        data[f"{label}_sam3_metadata_exists"] = metadata_path.exists()
        data[f"{label}_sam3_instance_count"] = len(list(instances_dir.glob("person_*.png"))) if instances_dir.exists() else 0
        if metadata_path.exists():
            artifacts.append(str(metadata_path))
        if instances_dir.exists():
            artifacts.append(str(instances_dir))

    status = "success" if not [w for w in warnings if w.startswith("missing")] else "failed"
    summary = (
        f"group persons={data.get('group_person_count', 0)}, "
        f"person records={data.get('person_person_count', 0)}, "
        f"group masks={data.get('group_sam3_instance_count', 0)}, "
        f"person masks={data.get('person_sam3_instance_count', 0)}"
    )
    return Observation(status=status, summary=summary, artifacts=artifacts, data=data, warnings=warnings)
