"""Stage 3 light-consistency and compositing tools exposed to the Agent."""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from ..adapters.person_inserter_adapter import load_candidate_patch
from ..schemas import CaseConfig, Observation
from ..tool_runner import run_command


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
        source_face_rgb = source[mask].reshape(-1, 1, 3)
        target_face_rgb = [target[mask].reshape(-1, 1, 3)]
        compose_stdout = StringIO()
        with redirect_stdout(compose_stdout):
            result = compositor.compose(
                source,
                target,
                mask,
                boundary,
                source_face_rgb=source_face_rgb,
                target_face_rgb=target_face_rgb,
            )
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
            "source_face_samples": int(source_face_rgb.shape[0]),
            "target_face_sample_sets": len(target_face_rgb),
            "output_image": str(out_img),
            "mrf_report": dict(getattr(compositor, "last_report", {}) or {}),
            "compose_stdout": compose_stdout.getvalue().strip(),
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
    preserve_detail: bool = True,
    feather_px: float = 2.0,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    use_active_solve: bool = True,
    **_: object,
) -> Observation:
    """Run MRF final compositing for one stage-2 insertion candidate."""
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
                    "stage2_summary": str(output_dir / "insertion" / "candidate_summaries.json"),
                    "margin": margin,
                    "max_crop_size": max_crop_size,
                    "preserve_detail": preserve_detail,
                    "feather_px": feather_px,
                    "max_iter": max_iter,
                    "tolerance": tolerance,
                    "use_active_solve": use_active_solve,
                }
            },
        )

    try:
        import numpy as np
        from PIL import Image, ImageOps

        ImageCompositor, PersonInserter = _import_current_modules(paths.cg_root)
        summary_path = output_dir / "insertion" / "candidate_summaries.json"
        stage2_candidate_summary: dict[str, Any] | None = None
        if summary_path.exists():
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summaries, list) and len(summaries) >= candidate_rank:
                item = summaries[candidate_rank - 1]
                if isinstance(item, dict):
                    stage2_candidate_summary = item
        group_rgb = np.asarray(
            ImageOps.exif_transpose(Image.open(paths.group_image)).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        patch_artifact = None
        if stage2_candidate_summary is not None:
            raw_patch_artifact = stage2_candidate_summary.get("patch_artifact")
            if isinstance(raw_patch_artifact, str) and raw_patch_artifact:
                patch_artifact = Path(raw_patch_artifact)
        if patch_artifact is not None and patch_artifact.exists():
            candidate = load_candidate_patch(patch_artifact, group_rgb, PersonInserter)
            loaded_from_cache = True
        else:
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
            loaded_from_cache = False
        compositor = ImageCompositor.MRFImageCompositor(max_iter=max_iter, tolerance=tolerance)
        compose_stdout = StringIO()
        with redirect_stdout(compose_stdout):
            result = PersonInserter.compose_and_paste(
                group_rgb,
                candidate,
                margin=margin,
                compositor=compositor,
                max_crop_size=max_crop_size,
                preserve_detail=preserve_detail,
                feather_px=feather_px,
                use_active_solve=use_active_solve,
            )
        final_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(final_img, quality=90)
        report = {
            "status": "success",
            "candidate_rank": candidate_rank,
            "stage2_summary": str(summary_path),
            "used_stage2_summary": stage2_candidate_summary is not None,
            "loaded_candidate_patch_cache": loaded_from_cache,
            "patch_artifact": str(patch_artifact) if patch_artifact is not None else "",
            "score": float(candidate.score),
            "scale": float(candidate.scale),
            "gap_bbox": [int(v) for v in candidate.gap_bbox],
            "margin": margin,
            "max_crop_size": max_crop_size,
            "preserve_detail": preserve_detail,
            "feather_px": feather_px,
            "max_iter": max_iter,
            "tolerance": tolerance,
            "use_active_solve": use_active_solve,
            "mrf_report": dict(getattr(compositor, "last_report", {}) or {}),
            "final_image": str(final_img),
            "compose_stdout": compose_stdout.getvalue().strip(),
        }
        target_face_bbox = getattr(candidate, "target_face_bbox", None)
        if target_face_bbox is not None:
            report["target_face_bbox"] = [int(v) for v in target_face_bbox]
        source_face_rgb = getattr(candidate, "source_face_rgb", None)
        target_face_rgb = getattr(candidate, "target_face_rgb", None)
        if source_face_rgb is not None:
            report["source_face_samples"] = int(getattr(source_face_rgb, "size", 0) // 3)
        if target_face_rgb is not None:
            report["target_face_sample_sets"] = len(target_face_rgb)
        if stage2_candidate_summary is not None:
            report["stage2_candidate"] = stage2_candidate_summary
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return Observation(
            status="success",
            summary=f"Composed candidate #{candidate_rank} with ImageCompositor.",
            artifacts=[str(final_img), str(report_path)],
            data=report,
        )
    except Exception as exc:
        return Observation(status="failed", summary=f"compose_top_candidate failed: {exc!r}")


def compose_all_candidates(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    candidate_count: int | None = None,
    **params: object,
) -> Observation:
    """Run MRF final compositing for every serialized stage-2 insertion candidate."""
    summary_path = output_dir / "insertion" / "candidate_summaries.json"
    final_dir = output_dir / "final"
    if candidate_count is None:
        candidate_count = case.top_k
        if summary_path.exists():
            try:
                summaries = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(summaries, list):
                    candidate_count = len(summaries)
            except Exception:
                candidate_count = case.top_k
    candidate_count = max(0, int(candidate_count))

    if dry_run:
        planned = []
        for rank in range(1, candidate_count + 1):
            planned.append({
                "candidate_rank": rank,
                "final_image": str(final_dir / f"{case.group_id}_{case.person_id}_candidate_{rank}.jpg"),
                "report": str(final_dir / f"{case.group_id}_{case.person_id}_candidate_{rank}_report.json"),
            })
        return Observation(
            status="skipped",
            summary=f"Dry run: all-candidate compositing planned for {candidate_count} candidate(s).",
            artifacts=[item["final_image"] for item in planned] + [item["report"] for item in planned],
            data={"planned_candidates": planned, "candidate_count": candidate_count, "params": dict(params)},
        )

    if candidate_count <= 0:
        return Observation(
            status="failed",
            summary="compose_all_candidates failed: no candidate summaries available",
            data={"stage2_summary": str(summary_path), "candidate_count": candidate_count},
        )

    artifacts: list[str] = []
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rank in range(1, candidate_count + 1):
        rank_params = dict(params)
        rank_params["candidate_rank"] = rank
        obs = compose_top_candidate(case, output_dir, dry_run=False, **rank_params)
        artifacts.extend(obs.artifacts)
        result = {
            "candidate_rank": rank,
            "status": obs.status,
            "summary": obs.summary,
            "artifacts": obs.artifacts,
        }
        if obs.status == "success":
            result.update({
                "final_image": obs.data.get("final_image", ""),
                "report": obs.artifacts[1] if len(obs.artifacts) > 1 else "",
                "score": obs.data.get("score", 0.0),
                "scale": obs.data.get("scale", 0.0),
                "mrf_report": obs.data.get("mrf_report", {}),
            })
        else:
            failures.append(result)
        results.append(result)

    report_path = final_dir / f"{case.group_id}_{case.person_id}_all_candidates_report.json"
    report = {
        "status": "success" if not failures else "failed",
        "candidate_count": candidate_count,
        "composed_count": candidate_count - len(failures),
        "failed_count": len(failures),
        "stage2_summary": str(summary_path),
        "candidates": results,
    }
    final_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(report_path))

    if failures:
        return Observation(
            status="failed",
            summary=f"Composed {candidate_count - len(failures)}/{candidate_count} candidate(s); failures remain.",
            artifacts=artifacts,
            data=report,
        )
    return Observation(
        status="success",
        summary=f"Composed all {candidate_count} candidate(s) with ImageCompositor.",
        artifacts=artifacts,
        data=report,
    )


def align_tone_hsv(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    max_hue_shift_deg: float = 18.0,
    min_saturation_ratio: float = 0.75,
    max_saturation_ratio: float = 1.35,
    value_strength: float = 0.0,
    strength: float = 1.0,
    min_valid_pixels: int = 128,
    **_: object,
) -> Observation:
    """Run scripts/align_tone_hsv.py for the case person against masked group people."""
    paths = case.paths()
    tone_dir = output_dir / "tone"
    output_image = tone_dir / f"{case.group_id}_{case.person_id}_hsv_aligned.png"
    report_path = tone_dir / f"{case.group_id}_{case.person_id}_hsv_report.json"
    person_mask = paths.person_sam3_dir / "instances" / "person_001.png"
    group_mask_glob = paths.group_sam3_dir / "instances" / "person_*.png"

    command = [
        "python",
        "scripts/align_tone_hsv.py",
        "--source-image",
        str(paths.person_image.relative_to(paths.cg_root)),
        "--source-mask",
        str(person_mask.relative_to(paths.cg_root)),
        "--target-image",
        str(paths.group_image.relative_to(paths.cg_root)),
        "--target-mask-glob",
        str(group_mask_glob.relative_to(paths.cg_root)),
        "--output-image",
        str(output_image),
        "--report",
        str(report_path),
        "--max-hue-shift-deg",
        str(max_hue_shift_deg),
        "--min-saturation-ratio",
        str(min_saturation_ratio),
        "--max-saturation-ratio",
        str(max_saturation_ratio),
        "--value-strength",
        str(value_strength),
        "--strength",
        str(strength),
        "--min-valid-pixels",
        str(min_valid_pixels),
    ]

    obs = run_command(command, cwd=paths.cg_root, dry_run=dry_run, timeout_s=300)
    obs.artifacts.extend([str(output_image), str(report_path)])
    obs.data.update({
        "stage": "tone_alignment",
        "method": "hsv",
        "output_image": str(output_image),
        "report": str(report_path),
        "source_image": str(paths.person_image),
        "source_mask": str(person_mask),
        "target_image": str(paths.group_image),
        "target_mask_glob": str(group_mask_glob),
    })
    if dry_run or obs.status != "success" or not report_path.exists():
        return obs

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        obs.data.update(report)
        correction = report.get("correction", {})
        hue = correction.get("applied_hue_shift_deg", 0.0)
        sat = correction.get("applied_saturation_ratio", 1.0)
        warnings = report.get("warnings", [])
        obs.summary = f"HSV tone alignment completed: hue_shift={hue}°, saturation_ratio={sat}."
        if warnings:
            obs.warnings.extend(str(w) for w in warnings)
    except Exception as exc:
        obs.warnings.append(f"could not parse HSV tone report: {exc!r}")
    return obs
