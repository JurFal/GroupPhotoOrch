"""Small deterministic revision policy for P0 ReAct traces."""
from __future__ import annotations

from typing import Any

from ..schemas import Decision, Verification


def decide_after_verification(stage: str, verification: Verification, pass_next: str) -> Decision:
    if verification.status == "pass":
        return Decision(next=pass_next, reason=f"{stage} verifier passed.")
    if verification.status == "skipped":
        return Decision(next=pass_next, reason=f"{stage} was skipped in dry-run mode.")
    if verification.status == "needs_revision" and verification.recommended_actions:
        return Decision(
            next="revise_or_stop",
            reason=f"{stage} verifier requested revision.",
            revision=verification.recommended_actions[0],
        )
    return Decision(next="stop", reason=f"{stage} verifier failed without a safe revision.")


def revision_limit_for(stage: str, limits: dict[str, int]) -> int:
    """Resolve per-stage retry limits while accepting old config aliases."""
    aliases = {
        "tone_alignment": "compositing",
        "lighting_plan": "compositing",
        "mrf_compositing": "compositing",
    }
    key = aliases.get(stage, stage)
    return int(limits.get(stage, limits.get(key, 1)))


def build_revision_params(
    stage: str,
    tool_name: str,
    current_params: dict[str, Any],
    verification: Verification,
    attempt: int,
) -> dict[str, Any] | None:
    """Convert verifier recommendations into safe tool-parameter retries."""
    params = dict(current_params)
    actions = verification.recommended_actions
    action_names = {
        str(action.get("action", ""))
        for action in actions
        if isinstance(action, dict)
    }

    for action in actions:
        if not isinstance(action, dict):
            continue
        nested = action.get("params")
        if isinstance(nested, dict):
            params.update(nested)

    if stage == "insertion" and tool_name == "geometry.find_insertion_candidates":
        if "use_candidate_rank" in action_names:
            return None
        if action_names & {
            "relax_gap_or_row_policy",
            "choose_next_candidate",
            "recompute_candidate_patch",
            "recompute_candidate_contour",
        }:
            current_top_k = int(params.get("top_k", 0) or 0)
            params["top_k"] = max(current_top_k + 3, 5 + attempt * 3)
            return params

    if stage == "tone_alignment" and tool_name == "compositing.align_tone_hsv":
        if action_names & {"clamp_hue_shift", "clamp_saturation_ratio"}:
            params["max_hue_shift_deg"] = min(float(params.get("max_hue_shift_deg", 18.0)), 12.0)
            params["min_saturation_ratio"] = max(float(params.get("min_saturation_ratio", 0.75)), 0.85)
            params["max_saturation_ratio"] = min(float(params.get("max_saturation_ratio", 1.35)), 1.20)
            params["strength"] = min(float(params.get("strength", 1.0)), 0.75)
            return params
        if action_names & {"use_more_or_better_masks", "check_individual_mask", "check_group_person_masks"}:
            params["min_valid_pixels"] = max(32, int(params.get("min_valid_pixels", 128)) // 2)
            params["strength"] = min(float(params.get("strength", 1.0)), 0.85)
            return params

    if stage == "mrf_compositing" and tool_name.startswith("compositing.compose_"):
        if action_names & {"lower_mrf_budget_or_choose_next_candidate", "tighten_crop_or_mask_to_reduce_active_area"}:
            params["max_crop_size"] = max(96, int(params.get("max_crop_size", 160)) - 24)
            params["feather_px"] = max(1.0, float(params.get("feather_px", 2.0)) - 0.5)
            return params
        if "enable_use_active_solve" in action_names:
            params["use_active_solve"] = True
            return params
        if "allow_more_iterations_if_quality_is_insufficient" in action_names:
            params["max_iter"] = min(300, int(params.get("max_iter", 80)) + 40)
            return params

    return None
