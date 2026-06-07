"""Verifier for HSV tone-alignment reports."""
from __future__ import annotations

from ..schemas import Observation, Verification


def verify_hsv_tone_alignment(observation: Observation) -> Verification:
    if observation.status == "skipped":
        return Verification(stage="tone_alignment", status="skipped", score=0.0)

    if observation.status != "success":
        return Verification(
            stage="tone_alignment",
            status="fail",
            score=0.0,
            problems=[observation.summary],
            recommended_actions=[{"action": "inspect_align_tone_hsv_inputs"}],
        )

    problems: list[str] = []
    actions: list[dict] = []
    data = observation.data
    source_stats = data.get("source_stats", {}) if isinstance(data.get("source_stats", {}), dict) else {}
    target_stats = data.get("target_stats", {}) if isinstance(data.get("target_stats", {}), dict) else {}
    correction = data.get("correction", {}) if isinstance(data.get("correction", {}), dict) else {}
    limits = data.get("limits", {}) if isinstance(data.get("limits", {}), dict) else {}

    source_valid = int(source_stats.get("valid_sv_pixels", 0))
    target_valid = int(target_stats.get("valid_sv_pixels", 0))
    min_valid = int(limits.get("min_valid_pixels", 128))
    if source_valid < min_valid:
        problems.append(f"source valid HSV pixels {source_valid} < minimum {min_valid}")
        actions.append({"action": "check_individual_mask"})
    if target_valid < min_valid:
        problems.append(f"target valid HSV pixels {target_valid} < minimum {min_valid}")
        actions.append({"action": "check_group_person_masks"})

    applied_hue = abs(float(correction.get("applied_hue_shift_deg", 0.0)))
    max_hue = float(limits.get("max_hue_shift_deg", 18.0))
    if applied_hue > max_hue + 1e-6:
        problems.append(f"applied hue shift {applied_hue:.3f} exceeds configured limit {max_hue:.3f}")
        actions.append({"action": "clamp_hue_shift"})

    sat_ratio = float(correction.get("applied_saturation_ratio", 1.0))
    sat_limits = limits.get("saturation_ratio", [0.75, 1.35])
    if not isinstance(sat_limits, list) or len(sat_limits) != 2:
        sat_limits = [0.75, 1.35]
    sat_low, sat_high = float(sat_limits[0]), float(sat_limits[1])
    if not sat_low <= sat_ratio <= sat_high:
        problems.append(f"applied saturation ratio {sat_ratio:.3f} outside [{sat_low:.3f}, {sat_high:.3f}]")
        actions.append({"action": "clamp_saturation_ratio"})

    output_image = str(data.get("output_image") or data.get("outputs", {}).get("output_image", ""))
    report = str(data.get("report") or data.get("outputs", {}).get("report", ""))
    if not output_image:
        problems.append("tone alignment output image missing from observation data")
    if not report:
        problems.append("tone alignment report missing from observation data")

    warnings = [str(w) for w in data.get("warnings", [])] + observation.warnings
    for warning in warnings:
        if "too few" in warning.lower():
            problems.append(warning)
            actions.append({"action": "use_more_or_better_masks"})

    score = 1.0 if not problems else max(0.0, 1.0 - 0.2 * len(problems))
    status = "pass" if not problems else "needs_revision"
    return Verification(
        stage="tone_alignment",
        status=status,
        score=round(score, 3),
        problems=problems,
        recommended_actions=actions,
    )
