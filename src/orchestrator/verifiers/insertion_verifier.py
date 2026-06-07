"""Verifier for PersonInserter candidate summaries."""
from __future__ import annotations

from ..schemas import Observation, Verification


def verify_candidate_summaries(observation: Observation) -> Verification:
    if observation.status == "skipped":
        return Verification(stage="insertion", status="skipped", score=0.0)
    problems: list[str] = []
    actions: list[dict] = []

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
        problems.append("no insertion candidates returned")
        actions.append({"action": "relax_gap_or_row_policy"})
    else:
        top = candidates[0]
        if float(top.get("score", 0.0)) <= 0:
            problems.append("top candidate score is not positive")
        scale = float(top.get("scale", 0.0))
        if not 0.15 <= scale <= 3.0:
            problems.append(f"top candidate scale {scale:.3f} is outside loose sanity range [0.15, 3.0]")
            actions.append({"action": "clamp_or_recompute_scale"})
        if int(top.get("refined_area", 0)) <= 0:
            problems.append("top candidate refined area is empty")
            actions.append({"action": "choose_next_candidate"})

    score = 1.0 if not problems else max(0.0, 1.0 - 0.25 * len(problems))
    status = "pass" if not problems else "needs_revision"
    return Verification(
        stage="insertion",
        status=status,
        score=round(score, 3),
        problems=problems,
        recommended_actions=actions,
    )
