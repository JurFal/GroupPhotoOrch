"""Shared data structures for the ReAct orchestrator."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["pending", "success", "failed", "skipped"]
VerifyStatus = Literal["pass", "needs_revision", "fail", "skipped"]


@dataclass
class ToolAction:
    tool: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    status: Status
    summary: str
    artifacts: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Verification:
    stage: str
    status: VerifyStatus
    score: float = 0.0
    problems: list[str] = field(default_factory=list)
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Decision:
    next: str
    reason: str
    revision: dict[str, Any] | None = None


@dataclass
class TraceStep:
    index: int
    thought: str
    action: ToolAction
    observation: Observation
    verification: Verification
    decision: Decision


@dataclass
class CasePaths:
    cg_root: Path
    group_image: Path
    person_image: Path
    group_meta: Path
    person_meta: Path
    group_sam3_dir: Path
    person_sam3_dir: Path

    def to_strings(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class CaseConfig:
    case_id: str
    goal: str
    computer_graphics_root: Path
    group_id: str
    person_id: str
    top_k: int = 5
    expected: dict[str, Any] = field(default_factory=dict)
    revision_limits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> "CaseConfig":
        root = Path(data.get("computer_graphics_root", "."))
        if not root.is_absolute():
            root = base_dir / root
        return cls(
            case_id=data["case_id"],
            goal=data.get("goal", "insert_person_into_group_photo"),
            computer_graphics_root=root,
            group_id=data["group_id"],
            person_id=data["person_id"],
            top_k=int(data.get("top_k", 5)),
            expected=dict(data.get("expected", {})),
            revision_limits=dict(data.get("revision_limits", {})),
        )

    def paths(self) -> CasePaths:
        root = self.computer_graphics_root
        return CasePaths(
            cg_root=root,
            group_image=root / "material" / f"{self.group_id}.jpg",
            person_image=root / "material" / f"{self.person_id}.jpg",
            group_meta=root / "person_metadata" / f"{self.group_id}.json",
            person_meta=root / "person_metadata" / f"{self.person_id}.json",
            group_sam3_dir=root / "sam3_masks" / self.group_id,
            person_sam3_dir=root / "sam3_masks" / self.person_id,
        )


@dataclass
class AgentTrace:
    case_id: str
    goal: str
    inputs: dict[str, str]
    dry_run: bool
    steps: list[TraceStep] = field(default_factory=list)
    final_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
