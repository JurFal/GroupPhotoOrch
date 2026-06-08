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
    summary = {
        "rank": rank,
        "score": float(candidate.score),
        "scale": float(candidate.scale),
        "offset_yx": [int(candidate.offset[0]), int(candidate.offset[1])],
        "gap_bbox_xyxy": [int(v) for v in candidate.gap_bbox],
        "neighbors": [int(v) for v in candidate.neighbors],
        "contour_points": int(len(candidate.contour)),
        "refined_area": int(candidate.refined_mask.sum()),
        "source_mask_area": int(candidate.source_mask.sum()),
        "original_source_mask_area": int(getattr(candidate, "original_source_mask_area", candidate.source_mask.sum())),
        "occluded_source_pixels": int(getattr(candidate, "occluded_source_pixels", 0)),
        "occlusion_ratio": float(getattr(candidate, "occlusion_ratio", 0.0)),
        "side_score": float(getattr(candidate, "side_score", 0.0)),
        "source_size_hw": [int(candidate.source_rgb.shape[0]), int(candidate.source_rgb.shape[1])],
        "warnings": list(getattr(candidate, "warnings", [])),
    }
    target_face_bbox = getattr(candidate, "target_face_bbox", None)
    if target_face_bbox is not None:
        summary["target_face_bbox_xywh"] = [int(v) for v in target_face_bbox]
    source_face_rgb = getattr(candidate, "source_face_rgb", None)
    if source_face_rgb is not None:
        summary["source_face_samples"] = int(getattr(source_face_rgb, "size", 0) // 3)
    target_face_rgb = getattr(candidate, "target_face_rgb", None)
    if target_face_rgb is not None:
        summary["target_face_sample_sets"] = len(target_face_rgb)
        summary["target_face_samples"] = int(sum(getattr(x, "size", 0) // 3 for x in target_face_rgb))
    return summary


def _save_candidate_patch(candidate: Any, rank: int, patch_path: Path) -> None:
    """Persist the heavy CandidatePatch arrays so stage 3 does not rerun stage 2."""
    import numpy as np

    def rgb_to_u8(rgb: Any) -> Any:
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    patch_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "rank": rank,
        "offset": [int(candidate.offset[0]), int(candidate.offset[1])],
        "scale": float(candidate.scale),
        "gap_bbox": [int(v) for v in candidate.gap_bbox],
        "score": float(candidate.score),
        "neighbors": [int(v) for v in candidate.neighbors],
        "warnings": list(getattr(candidate, "warnings", [])),
        "original_source_mask_area": int(getattr(candidate, "original_source_mask_area", candidate.source_mask.sum())),
        "occluded_source_pixels": int(getattr(candidate, "occluded_source_pixels", 0)),
        "occlusion_ratio": float(getattr(candidate, "occlusion_ratio", 0.0)),
        "side_score": float(getattr(candidate, "side_score", 0.0)),
        "target_face_rgb_count": len(getattr(candidate, "target_face_rgb", [])),
        "source_mask_shape": list(candidate.source_mask.shape),
        "refined_mask_shape": list(candidate.refined_mask.shape),
        "encoding": "rgb_u8_packbits_v1",
    }
    arrays: dict[str, Any] = {
        "meta_json": np.array(json.dumps(meta, ensure_ascii=False)),
        "source_rgb_u8": rgb_to_u8(candidate.source_rgb),
        "source_mask_packed": np.packbits(candidate.source_mask.reshape(-1).astype(np.uint8)),
        "refined_mask_packed": np.packbits(candidate.refined_mask.reshape(-1).astype(np.uint8)),
        "contour": candidate.contour.astype(np.int32, copy=False),
        "source_face_rgb_u8": rgb_to_u8(candidate.source_face_rgb),
    }
    for idx, face_rgb in enumerate(getattr(candidate, "target_face_rgb", [])):
        arrays[f"target_face_rgb_u8_{idx:03d}"] = rgb_to_u8(face_rgb)
    # Use uncompressed NPZ: larger on disk, much faster to write/read during agent loops.
    np.savez(patch_path, **arrays)


def load_candidate_patch(patch_path: Path, target_rgb: Any, person_inserter: Any) -> Any:
    """Load a cached CandidatePatch NPZ artifact."""
    import numpy as np

    def u8_to_rgb(arr: Any) -> Any:
        return arr.astype(np.float32) / 255.0

    def unpack_mask(data: Any, key: str, shape: list[int]) -> Any:
        total = int(shape[0]) * int(shape[1])
        unpacked = np.unpackbits(data[key])[:total]
        return unpacked.reshape((int(shape[0]), int(shape[1]))).astype(bool, copy=False)

    with np.load(patch_path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        if str(meta.get("encoding", "")) == "rgb_u8_packbits_v1":
            source_rgb = u8_to_rgb(data["source_rgb_u8"])
            source_mask = unpack_mask(data, "source_mask_packed", meta["source_mask_shape"])
            refined_mask = unpack_mask(data, "refined_mask_packed", meta["refined_mask_shape"])
            source_face_rgb = u8_to_rgb(data["source_face_rgb_u8"])
            target_face_rgb = [
                u8_to_rgb(data[f"target_face_rgb_u8_{idx:03d}"])
                for idx in range(int(meta.get("target_face_rgb_count", 0)))
            ]
        else:
            # Backward compatibility with early cache artifacts.
            source_rgb = data["source_rgb"].astype(np.float32, copy=False)
            source_mask = data["source_mask"].astype(bool, copy=False)
            refined_mask = data["refined_mask"].astype(bool, copy=False)
            source_face_rgb = data["source_face_rgb"].astype(np.float32, copy=False)
            target_face_rgb = [
                data[f"target_face_rgb_{idx:03d}"].astype(np.float32, copy=False)
                for idx in range(int(meta.get("target_face_rgb_count", 0)))
            ]
        return person_inserter.CandidatePatch(
            target_rgb=target_rgb,
            source_rgb=source_rgb,
            target_face_rgb=target_face_rgb,
            source_face_rgb=source_face_rgb,
            source_mask=source_mask,
            refined_mask=refined_mask,
            offset=(int(meta["offset"][0]), int(meta["offset"][1])),
            contour=data["contour"].astype(np.int32, copy=False),
            scale=float(meta["scale"]),
            gap_bbox=[int(v) for v in meta["gap_bbox"]],
            score=float(meta["score"]),
            neighbors=[int(v) for v in meta.get("neighbors", [])],
            original_source_mask_area=int(meta.get("original_source_mask_area", source_mask.sum())),
            occluded_source_pixels=int(meta.get("occluded_source_pixels", 0)),
            occlusion_ratio=float(meta.get("occlusion_ratio", 0.0)),
            side_score=float(meta.get("side_score", 0.0)),
            warnings=list(meta.get("warnings", [])),
        )


def find_candidates(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    top_k: int | None = None,
    **_: object,
) -> Observation:
    paths = case.paths()
    summary_path = output_dir / "insertion" / "candidate_summaries.json"
    patch_dir = output_dir / "insertion" / "patches"
    requested_top_k = int(top_k or case.top_k)

    if dry_run:
        planned = {
            "group_meta_path": str(paths.group_meta),
            "group_image_path": str(paths.group_image),
            "individual_meta_path": str(paths.person_meta),
            "individual_image_path": str(paths.person_image),
            "top_k": requested_top_k,
            "patch_dir": str(patch_dir),
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
            top_k=requested_top_k,
        )
    except Exception as exc:
        return Observation(status="failed", summary=f"find_insertion_patches failed: {exc!r}")

    summaries = []
    patch_paths: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        patch_path = patch_dir / f"candidate_{idx:03d}.npz"
        _save_candidate_patch(candidate, idx, patch_path)
        patch_paths.append(str(patch_path))
        summary = _candidate_summary(candidate, idx)
        summary["patch_artifact"] = str(patch_path)
        summaries.append(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    return Observation(
        status="success",
        summary=f"Found {len(summaries)} insertion candidate(s).",
        artifacts=[str(summary_path), *patch_paths],
        data={"candidate_count": len(summaries), "candidates": summaries, "patch_dir": str(patch_dir)},
    )
