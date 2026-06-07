"""Stage 3 light-consistency and compositing tools exposed to the Agent."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..schemas import CaseConfig, Observation


def _import_current_modules(cg_root: Path):
    root_str = str(cg_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import ImageCompositor  # type: ignore
    import PersonInserter  # type: ignore

    return ImageCompositor, PersonInserter


def run_light_smoke(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    size: int = 48,
    max_iter: int = 50,
    tolerance: float = 1e-4,
    **_: object,
) -> Observation:
    """Run the ImageCompositor MRF prototype on a tiny synthetic patch."""
    paths = case.paths()
    report_path = output_dir / "final" / "light_smoke_report.json"
    if dry_run:
        return Observation(
            status="skipped",
            summary="Dry run: ImageCompositor light smoke was not executed.",
            artifacts=[str(report_path)],
            data={"planned": {"size": size, "max_iter": max_iter, "tolerance": tolerance}},
        )

    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import binary_dilation

        ImageCompositor, _ = _import_current_modules(paths.cg_root)
        compositor = ImageCompositor.MRFImageCompositor(max_iter=max_iter, tolerance=tolerance)
        h = w = int(size)
        source = np.zeros((h, w, 3), dtype=np.float32)
        target = np.ones((h, w, 3), dtype=np.float32) * 0.78
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= (min(h, w) // 3) ** 2
        source[mask] = [0.32, 0.32, 0.32]
        source[~mask] = [1.0, 1.0, 1.0]
        boundary = binary_dilation(mask, iterations=1) & ~mask
        result = compositor.compose(source, target, mask, boundary)
        out_img = output_dir / "final" / "light_smoke.png"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(out_img)
        report = {
            "status": "success",
            "size": size,
            "max_iter": max_iter,
            "tolerance": tolerance,
            "foreground_area": int(mask.sum()),
            "boundary_area": int(boundary.sum()),
            "output_image": str(out_img),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return Observation(
            status="success",
            summary="ImageCompositor light smoke completed.",
            artifacts=[str(out_img), str(report_path)],
            data=report,
        )
    except Exception as exc:
        return Observation(status="failed", summary=f"light smoke failed: {exc!r}")


def compose_top_candidate(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    candidate_rank: int = 1,
    margin: int = 8,
    max_crop_size: int = 200,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    **_: object,
) -> Observation:
    """Find candidates and run PersonInserter.compose_and_paste on one candidate."""
    paths = case.paths()
    final_dir = output_dir / "final"
    final_img = final_dir / f"{case.group_id}_{case.person_id}_candidate_{candidate_rank}.jpg"
    report_path = final_dir / f"{case.group_id}_{case.person_id}_candidate_{candidate_rank}_report.json"
    if dry_run:
        return Observation(
            status="skipped",
            summary="Dry run: top-candidate compositing was not executed.",
            artifacts=[str(final_img), str(report_path)],
            data={
                "planned": {
                    "candidate_rank": candidate_rank,
                    "margin": margin,
                    "max_crop_size": max_crop_size,
                    "max_iter": max_iter,
                    "tolerance": tolerance,
                }
            },
        )

    try:
        import numpy as np
        from PIL import Image, ImageOps

        ImageCompositor, PersonInserter = _import_current_modules(paths.cg_root)
        candidates = PersonInserter.find_insertion_patches(
            group_meta_path=paths.group_meta,
            group_image_path=paths.group_image,
            individual_meta_path=paths.person_meta,
            individual_image_path=paths.person_image,
            top_k=max(candidate_rank, case.top_k),
        )
        if len(candidates) < candidate_rank:
            return Observation(status="failed", summary=f"candidate rank {candidate_rank} not available")
        candidate = candidates[candidate_rank - 1]
        group_rgb = np.asarray(
            ImageOps.exif_transpose(Image.open(paths.group_image)).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        compositor = ImageCompositor.MRFImageCompositor(max_iter=max_iter, tolerance=tolerance)
        result = PersonInserter.compose_and_paste(
            group_rgb,
            candidate,
            margin=margin,
            compositor=compositor,
            max_crop_size=max_crop_size,
        )
        final_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(final_img, quality=90)
        report = {
            "status": "success",
            "candidate_rank": candidate_rank,
            "score": float(candidate.score),
            "scale": float(candidate.scale),
            "gap_bbox": [int(v) for v in candidate.gap_bbox],
            "target_face_bbox": [int(v) for v in candidate.target_face_bbox],
            "margin": margin,
            "max_crop_size": max_crop_size,
            "max_iter": max_iter,
            "tolerance": tolerance,
            "final_image": str(final_img),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return Observation(
            status="success",
            summary=f"Composed candidate #{candidate_rank} with ImageCompositor.",
            artifacts=[str(final_img), str(report_path)],
            data=report,
        )
    except Exception as exc:
        return Observation(status="failed", summary=f"compose_top_candidate failed: {exc!r}")
