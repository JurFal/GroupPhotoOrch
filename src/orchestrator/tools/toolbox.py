"""Callable tool dispatcher for the ReAct orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..schemas import CaseConfig, Observation
from . import stage1_tools, stage2_tools, stage3_tools, vision_tools

ToolFn = Callable[..., Observation]


class ToolBox:
    """Small whitelist dispatcher; tools are addressed by registry names."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {
            "vision.inspect_existing_artifacts": vision_tools.inspect_existing_artifacts,
            "vision.generate_sam3_masks": stage1_tools.generate_sam3_masks,
            "vision.generate_yolo_person_masks": stage1_tools.generate_yolo_person_masks,
            "vision.extract_metadata_from_masks": stage1_tools.extract_metadata_from_masks,
            "geometry.find_insertion_candidates": stage2_tools.find_insertion_candidates,
            "compositing.align_tone_hsv": stage3_tools.align_tone_hsv,
            "compositing.run_light_smoke": stage3_tools.run_light_smoke,
            "compositing.compose_top_candidate": stage3_tools.compose_top_candidate,
        }

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(
        self,
        name: str,
        case: CaseConfig,
        output_dir: Path,
        dry_run: bool = False,
        **params: Any,
    ) -> Observation:
        if name not in self._tools:
            return Observation(status="failed", summary=f"unknown tool: {name}")
        return self._tools[name](case=case, output_dir=output_dir, dry_run=dry_run, **params)
