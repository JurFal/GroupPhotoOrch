"""Small deterministic revision policy for P0 ReAct traces."""
from __future__ import annotations

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
