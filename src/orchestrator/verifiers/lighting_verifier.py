"""Verifier and deterministic MRF-budget policy for light-consistent compositing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import Observation, Verification


def choose_mrf_params(
    tone_observation: Observation,
    candidate_observation: Observation | None = None,
    *,
    candidate_rank: int = 1,
) -> dict[str, Any]:
    """Pick a small but useful MRF iteration budget from tone/candidate evidence."""
    data = tone_observation.data
    correction = data.get("correction", {}) if isinstance(data.get("correction", {}), dict) else {}
    hue_shift = abs(float(correction.get("applied_hue_shift_deg", 0.0)))
    sat_ratio = float(correction.get("applied_saturation_ratio", 1.0))
    sat_delta = abs(sat_ratio - 1.0)
    tone_severity = hue_shift / 18.0 + sat_delta / 0.35

    refined_area = 0
    source_h = 0
    source_w = 0
    if candidate_observation is not None:
        candidates = candidate_observation.data.get("candidates", [])
        if isinstance(candidates, list) and len(candidates) >= candidate_rank:
            top = candidates[candidate_rank - 1]
            if isinstance(top, dict):
                refined_area = int(top.get("refined_area", 0))
                size = top.get("source_size_hw", [0, 0])
                if isinstance(size, list) and len(size) == 2:
                    source_h, source_w = int(size[0]), int(size[1])

    if tone_severity >= 1.0 or refined_area >= 30000:
        max_iter = 80
        max_crop_size = 180
    elif tone_severity >= 0.45 or refined_area >= 12000:
        max_iter = 50
        max_crop_size = 160
    else:
        max_iter = 24
        max_crop_size = 140

    if max(source_h, source_w) > 900:
        max_crop_size = min(max_crop_size, 150)

    return {
        "candidate_rank": candidate_rank,
        "max_iter": max_iter,
        "max_crop_size": max_crop_size,
        "tolerance": 1e-4,
        "preserve_detail": True,
        "feather_px": 2.0,
        "use_active_solve": True,
    }


def verify_lighting_plan(
    tone_observation: Observation,
    candidate_observation: Observation | None = None,
) -> Verification:
    """Require a final MRF pass even after HSV succeeds, with an adaptive budget."""
    if tone_observation.status not in {"success", "skipped"}:
        return Verification(
            stage="lighting_plan",
            status="fail",
            score=0.0,
            problems=[tone_observation.summary],
            recommended_actions=[{"action": "retry_hsv_before_mrf"}],
        )

    params = choose_mrf_params(tone_observation, candidate_observation)
    action = {"action": "compositing.compose_top_candidate", "params": params}
    return Verification(
        stage="lighting_plan",
        status="pass",
        score=1.0,
        recommended_actions=[action],
    )


def verify_mrf_composite(observation: Observation) -> Verification:
    if observation.status == "skipped":
        return Verification(stage="mrf_compositing", status="skipped", score=0.0)
    if observation.status != "success":
        return Verification(
            stage="mrf_compositing",
            status="fail",
            score=0.0,
            problems=[observation.summary],
            recommended_actions=[{"action": "lower_mrf_budget_or_choose_next_candidate"}],
        )

    problems: list[str] = []
    actions: list[dict[str, Any]] = []
    data = observation.data
    final_image = str(data.get("final_image", ""))
    if not final_image or not Path(final_image).exists():
        problems.append("final MRF composite image is missing")
        actions.append({"action": "rerun_compositing.compose_top_candidate"})

    report = data.get("mrf_report", {}) if isinstance(data.get("mrf_report", {}), dict) else {}
    if report:
        active_ratio = float(report.get("active_ratio", 1.0))
        actual_iterations = int(report.get("actual_iterations", 0))
        max_iter = int(report.get("max_iter", data.get("max_iter", 0)))
        if str(report.get("solve_mode", "")) != "active":
            actions.append({"action": "enable_use_active_solve"})
        if active_ratio > 0.75:
            actions.append({"action": "tighten_crop_or_mask_to_reduce_active_area"})
        if max_iter > 0 and actual_iterations >= max_iter:
            actions.append({"action": "allow_more_iterations_if_quality_is_insufficient"})
    else:
        actions.append({"action": "expose_mrf_report_metadata"})

    score = 1.0 if not problems else max(0.0, 1.0 - 0.3 * len(problems))
    return Verification(
        stage="mrf_compositing",
        status="pass" if not problems else "needs_revision",
        score=round(score, 3),
        problems=problems,
        recommended_actions=actions,
    )
