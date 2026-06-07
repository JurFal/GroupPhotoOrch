"""Stage 1 perception tools exposed to the Agent."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..schemas import CaseConfig, Observation
from ..tool_runner import run_command

ImageSelection = Literal["group", "person", "both", "all_material"]


def _selected_images(case: CaseConfig, selection: ImageSelection) -> list[Path]:
    paths = case.paths()
    if selection == "group":
        return [paths.group_image]
    if selection == "person":
        return [paths.person_image]
    if selection == "both":
        return [paths.group_image, paths.person_image]
    return sorted((paths.cg_root / "material").glob("*.jpg"))


def _rel_paths(paths: list[Path], root: Path) -> list[str]:
    out = []
    for path in paths:
        try:
            out.append(str(path.relative_to(root)))
        except ValueError:
            out.append(str(path))
    return out


def generate_sam3_masks(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    image_selection: ImageSelection = "both",
    checkpoint: str = "",
    allow_hf_download: bool = False,
    confidence: float = 0.5,
    device: str = "cuda",
    max_inference_side: int = 1600,
    **_: object,
) -> Observation:
    """Call Computer-Graphics/scripts/generate_sam3_masks.py."""
    paths = case.paths()
    images = _selected_images(case, image_selection)
    out_dir = output_dir / "vision" / "sam3_masks"
    command = [
        "python",
        "scripts/generate_sam3_masks.py",
        *_rel_paths(images, paths.cg_root),
        "--output-dir",
        str(out_dir),
        "--confidence",
        str(confidence),
        "--device",
        device,
        "--max-inference-side",
        str(max_inference_side),
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if allow_hf_download:
        command.append("--allow-hf-download")
    obs = run_command(command, cwd=paths.cg_root, dry_run=dry_run, timeout_s=3600)
    obs.artifacts.append(str(out_dir))
    obs.data.update({"stage": "vision", "method": "sam3", "images": [str(p) for p in images]})
    return obs


def generate_yolo_person_masks(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    image_selection: ImageSelection = "both",
    model: str = "yolo11m-seg.pt",
    conf: float = 0.35,
    imgsz: int = 1024,
    device: str = "0",
    half: bool = True,
    auto_tile_size: int = 1280,
    tile_overlap: int = 256,
    **_: object,
) -> Observation:
    """Call Computer-Graphics/scripts/generate_yolo_person_masks.py."""
    paths = case.paths()
    images = _selected_images(case, image_selection)
    out_dir = output_dir / "vision" / "yolo_masks"
    command = [
        "python",
        "scripts/generate_yolo_person_masks.py",
        *_rel_paths(images, paths.cg_root),
        "--model",
        model,
        "--output-dir",
        str(out_dir),
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
        "--device",
        device,
        "--auto-tile-size",
        str(auto_tile_size),
        "--tile-overlap",
        str(tile_overlap),
    ]
    if half:
        command.append("--half")
    obs = run_command(command, cwd=paths.cg_root, dry_run=dry_run, timeout_s=3600)
    obs.artifacts.append(str(out_dir))
    obs.data.update({"stage": "vision", "method": "yolo", "images": [str(p) for p in images]})
    return obs


def extract_metadata_from_masks(
    case: CaseConfig,
    output_dir: Path,
    dry_run: bool = False,
    masks_dir: str | None = None,
    metadata_output_dir: str | None = None,
    contour_stride: int = 1,
    min_score: float = 0.0,
    min_area: int = 0,
    **_: object,
) -> Observation:
    """Call Computer-Graphics/scripts/extract_metadata_from_sam3.py on SAM3/YOLO layout."""
    paths = case.paths()
    mask_root = Path(masks_dir) if masks_dir else paths.cg_root / "sam3_masks"
    if not mask_root.is_absolute():
        mask_root = paths.cg_root / mask_root
    out_dir = Path(metadata_output_dir) if metadata_output_dir else output_dir / "vision" / "person_metadata"
    command = [
        "python",
        "scripts/extract_metadata_from_sam3.py",
        "--sam3-dir",
        str(mask_root),
        "--output-dir",
        str(out_dir),
        "--contour-stride",
        str(contour_stride),
        "--min-score",
        str(min_score),
        "--min-area",
        str(min_area),
    ]
    obs = run_command(command, cwd=paths.cg_root, dry_run=dry_run, timeout_s=1200)
    obs.artifacts.append(str(out_dir))
    obs.data.update({"stage": "vision", "method": "metadata_extraction", "masks_dir": str(mask_root)})
    return obs
