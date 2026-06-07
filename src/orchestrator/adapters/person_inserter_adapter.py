"""Adapter for the current Computer-Graphics/PersonInserter.py interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..schemas import CaseConfig, Observation


def _import_person_inserter(cg_root: Path):
    root_str = str(cg_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import PersonInserter  # type: ignore

    return PersonInserter


def _candidate_summary(candidate: Any, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": float(candidate.score),
        "scale": float(candidate.scale),
        "offset_yx": [int(candidate.offset[0]), int(candidate.offset[1])],
        "target_face_bbox_xywh": [int(v) for v in candidate.target_face_bbox],
        "gap_bbox_xyxy": [int(v) for v in candidate.gap_bbox],
        "neighbors": [int(v) for v in candidate.neighbors],
        "contour_points": int(len(candidate.contour)),
        "refined_area": int(candidate.refined_mask.sum()),
        "source_size_hw": [int(candidate.source_rgb.shape[0]), int(candidate.source_rgb.shape[1])],
        "warnings": list(getattr(candidate, "warnings", [])),
    }


def find_candidates(case: CaseConfig, output_dir: Path, dry_run: bool = False) -> Observation:
    paths = case.paths()
    summary_path = output_dir / "insertion" / "candidate_summaries.json"

    if dry_run:
        planned = {
            "group_meta_path": str(paths.group_meta),
            "group_image_path": str(paths.group_image),
            "individual_meta_path": str(paths.person_meta),
            "individual_image_path": str(paths.person_image),
            "top_k": case.top_k,
        }
        return Observation(
            status="skipped",
            summary="Dry run: PersonInserter.find_insertion_patches was not called.",
            artifacts=[],
            data={"planned_call": planned},
        )

    try:
        inserter = _import_person_inserter(paths.cg_root)
        candidates = inserter.find_insertion_patches(
            group_meta_path=paths.group_meta,
            group_image_path=paths.group_image,
            individual_meta_path=paths.person_meta,
            individual_image_path=paths.person_image,
            top_k=case.top_k,
        )
    except Exception as exc:
        return Observation(status="failed", summary=f"find_insertion_patches failed: {exc!r}")

    summaries = [_candidate_summary(candidate, idx) for idx, candidate in enumerate(candidates, start=1)]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    return Observation(
        status="success",
        summary=f"Found {len(summaries)} insertion candidate(s).",
        artifacts=[str(summary_path)],
        data={"candidate_count": len(summaries), "candidates": summaries},
    )
