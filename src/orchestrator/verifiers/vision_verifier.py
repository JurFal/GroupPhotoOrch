"""Verifier for raw Computer-Graphics vision artifacts."""
from __future__ import annotations

from ..schemas import CaseConfig, Observation, Verification


def verify_existing_artifacts(case: CaseConfig, observation: Observation) -> Verification:
    if observation.status == "skipped":
        return Verification(stage="vision", status="skipped", score=0.0)
    problems: list[str] = []
    actions: list[dict] = []
    data = observation.data
    expected = case.expected

    if observation.status != "success":
        problems.extend(observation.warnings or ["artifact inspection failed"])

    group_count = int(data.get("group_person_count", 0))
    person_count = int(data.get("person_person_count", 0))
    group_masks = int(data.get("group_sam3_instance_count", 0))
    person_masks = int(data.get("person_sam3_instance_count", 0))

    min_group = int(expected.get("min_group_persons", 1))
    max_group = int(expected.get("max_group_persons", 9999))
    min_individual = int(expected.get("min_persons_in_individual", 1))

    if group_count < min_group:
        problems.append(f"group person count {group_count} < expected minimum {min_group}")
        actions.append({"action": "regenerate_group_masks", "reason": "too few group persons"})
    if group_count > max_group:
        problems.append(f"group person count {group_count} > expected maximum {max_group}")
        actions.append({"action": "raise_detection_threshold", "reason": "too many group persons"})
    if person_count < min_individual:
        problems.append(f"individual person records {person_count} < expected minimum {min_individual}")
        actions.append({"action": "regenerate_individual_mask", "reason": "missing individual person metadata"})
    if group_masks <= 0:
        problems.append("no group SAM3 instance masks found")
    if person_masks <= 0:
        problems.append("no individual SAM3 instance masks found")

    score = 1.0
    if problems:
        score = max(0.0, 1.0 - 0.18 * len(problems))
    status = "pass" if not problems else ("needs_revision" if actions else "fail")
    return Verification(
        stage="vision",
        status=status,
        score=round(score, 3),
        problems=problems,
        recommended_actions=actions,
    )
