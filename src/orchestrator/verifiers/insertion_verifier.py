"""Verifier for PersonInserter candidate summaries."""
from __future__ import annotations

import math
from typing import Any

from ..schemas import Observation, Verification


def _positive_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number > 0 else default


def _candidate_is_composable(candidate: dict[str, Any]) -> tuple[bool, list[str], list[dict[str, Any]]]:
    problems: list[str] = []
    actions: list[dict[str, Any]] = []

    if _positive_number(candidate.get("score")) <= 0:
        problems.append("candidate score is not positive")
        actions.append({"action": "choose_next_candidate"})

    refined_area = int(_positive_number(candidate.get("refined_area")))
    if refined_area <= 0:
        problems.append("candidate refined area is empty")
        actions.append({"action": "choose_next_candidate"})

    source_size = candidate.get("source_size_hw", [0, 0])
    if not isinstance(source_size, list) or len(source_size) != 2:
        source_size = [0, 0]
    source_h = int(_positive_number(source_size[0] if source_size else 0))
    source_w = int(_positive_number(source_size[1] if len(source_size) > 1 else 0))
    if source_h <= 0 or source_w <= 0:
        problems.append("candidate source patch size is invalid")
        actions.append({"action": "recompute_candidate_patch"})

    contour_points = int(_positive_number(candidate.get("contour_points")))
    if contour_points <= 0:
        problems.append("candidate contour is empty")
        actions.append({"action": "recompute_candidate_contour"})

    original_area = _positive_number(candidate.get("original_source_mask_area"))
    visible_area = _positive_number(candidate.get("source_mask_area"), _positive_number(candidate.get("refined_area")))
    occluded_pixels = _positive_number(candidate.get("occluded_source_pixels"))
    occlusion_ratio = _positive_number(candidate.get("occlusion_ratio"), -1.0)
    if occlusion_ratio < 0 and original_area > 0:
        occlusion_ratio = max(0.0, min(1.0, occluded_pixels / original_area))
    if original_area > 0 and visible_area > original_area:
        problems.append("candidate visible mask area exceeds original source mask area")
        actions.append({"action": "recompute_occlusion_mask"})
    if occlusion_ratio >= 0.70:
        problems.append(f"candidate foreground occlusion ratio {occlusion_ratio:.3f} is too high")
        actions.append({"action": "prefer_side_or_less_occluded_gap"})
    elif occlusion_ratio >= 0.45:
        actions.append({"action": "review_occlusion_quality", "occlusion_ratio": round(occlusion_ratio, 3)})

    side_score = _positive_number(candidate.get("side_score"), -1.0)
    neighbors = candidate.get("neighbors", [])
    if side_score < 0.25 and isinstance(neighbors, list) and len(neighbors) >= 2:
        actions.append({"action": "prefer_group_edge_gap", "side_score": round(side_score, 3)})

    patch_artifact = candidate.get("patch_artifact")
    if patch_artifact is not None and not isinstance(patch_artifact, str):
        problems.append("candidate patch artifact path is invalid")
        actions.append({"action": "rewrite_candidate_patch_cache"})

    return not problems, problems, actions


def verify_candidate_summaries(observation: Observation) -> Verification:
    if observation.status == "skipped":
        return Verification(stage="insertion", status="skipped", score=0.0)

    if observation.status != "success":
        return Verification(
            stage="insertion",
            status="fail",
            score=0.0,
            problems=[observation.summary],
            recommended_actions=[{"action": "inspect_person_inserter_inputs"}],
        )

    candidates = observation.data.get("candidates", [])
    if not candidates:
        return Verification(
            stage="insertion",
            status="needs_revision",
            score=0.0,
            problems=["no insertion candidates returned"],
            recommended_actions=[{"action": "relax_gap_or_row_policy"}],
        )

    valid_rank = 0
    valid_candidate: dict[str, Any] | None = None
    valid_actions: list[dict[str, Any]] = []
    problems: list[str] = []
    actions: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            problems.append(f"candidate #{idx} summary is not an object")
            actions.append({"action": "rewrite_candidate_summary"})
            continue
        ok, candidate_problems, candidate_actions = _candidate_is_composable(candidate)
        if ok:
            valid_rank = idx
            valid_candidate = candidate
            valid_actions = candidate_actions
            break
        if idx == 1:
            problems.extend(f"top candidate {problem}" for problem in candidate_problems)
            actions.extend(candidate_actions)

    if valid_rank:
        occlusion_ratio = 0.0
        side_score = 1.0
        if valid_candidate is not None:
            occlusion_ratio = _positive_number(valid_candidate.get("occlusion_ratio"), 0.0)
            side_score = _positive_number(valid_candidate.get("side_score"), 1.0)
        score = 1.0 if valid_rank == 1 else 0.85
        score = max(0.65, score - 0.25 * occlusion_ratio - (0.05 if side_score < 0.25 else 0.0))
        recommended = list(valid_actions)
        if valid_rank != 1:
            recommended.insert(0, {"action": "use_candidate_rank", "params": {"candidate_rank": valid_rank}})
        return Verification(
            stage="insertion",
            status="pass",
            score=round(score, 3),
            problems=[] if valid_rank == 1 else [f"top candidate invalid; candidate #{valid_rank} is composable"],
            recommended_actions=recommended,
        )

    return Verification(
        stage="insertion",
        status="needs_revision",
        score=0.0,
        problems=problems or ["no composable insertion candidates returned"],
        recommended_actions=actions or [{"action": "relax_gap_or_row_policy"}],
    )
