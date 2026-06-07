"""Rule-based ReAct orchestrator for the current Computer-Graphics tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .env import load_project_env
from .llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleClient
from .policies.llm_decision_policy import choose_next_action
from .policies.revision_policy import decide_after_verification
from .schemas import AgentTrace, CaseConfig, Decision, Observation, ToolAction, TraceStep, Verification
from .trace import write_trace
from .tools.toolbox import ToolBox
from .verifiers.insertion_verifier import verify_candidate_summaries
from .verifiers.vision_verifier import verify_existing_artifacts


DEFAULT_LLM_ALLOWED_TOOLS = [
    "compositing.run_light_smoke",
    "compositing.compose_top_candidate",
    "vision.extract_metadata_from_masks",
    "vision.inspect_existing_artifacts",
]


class ReActOrchestrator:
    def __init__(self, project_root: Path, output_root: Path | None = None) -> None:
        self.project_root = project_root
        self.output_root = output_root or project_root / "orchestrator" / "outputs"
        self.tools = ToolBox()

    def run_case(
        self,
        case: CaseConfig,
        dry_run: bool = False,
        use_llm_decision: bool = False,
        execute_llm_tool: bool = False,
        llm_allowed_tools: list[str] | None = None,
        llm_base_url: str = DEFAULT_BASE_URL,
        llm_model: str = DEFAULT_MODEL,
    ) -> AgentTrace:
        case_output = self.output_root / case.case_id
        paths = case.paths()
        trace = AgentTrace(
            case_id=case.case_id,
            goal=case.goal,
            inputs={
                "group_image": str(paths.group_image),
                "person_image": str(paths.person_image),
                "group_meta": str(paths.group_meta),
                "person_meta": str(paths.person_meta),
            },
            dry_run=dry_run,
        )

        # Step 1: observe existing raw artifacts.
        obs = self.tools.call("vision.inspect_existing_artifacts", case, output_dir=case_output, dry_run=dry_run)
        ver = verify_existing_artifacts(case, obs)
        dec = decide_after_verification("vision", ver, pass_next="find_insertion_candidates")
        trace.steps.append(TraceStep(
            index=1,
            thought="Need to inspect existing SAM3/person_metadata artifacts before planning insertion.",
            action=ToolAction(tool="vision.inspect_existing_artifacts", params={"case_id": case.case_id}),
            observation=obs,
            verification=ver,
            decision=dec,
        ))
        write_trace(case_output / "agent_trace.json", trace)
        if dec.next in {"stop", "revise_or_stop"} and not dry_run:
            trace.final_status = "blocked_at_vision"
            write_trace(case_output / "agent_trace.json", trace)
            return trace

        # Step 2: find insertion candidates through PersonInserter adapter.
        obs2 = self.tools.call("insertion.find_candidates", case, output_dir=case_output, dry_run=dry_run)
        ver2 = verify_candidate_summaries(obs2)
        dec2 = decide_after_verification("insertion", ver2, pass_next="ready_for_optional_compositing")
        trace.steps.append(TraceStep(
            index=2,
            thought="Use the current PersonInserter interface to generate candidate patches; serialize only lightweight summaries.",
            action=ToolAction(tool="insertion.find_candidates", params={"top_k": case.top_k}),
            observation=obs2,
            verification=ver2,
            decision=dec2,
        ))

        if dec2.next == "ready_for_optional_compositing":
            trace.final_status = "success_dry_run" if dry_run else "candidates_ready"
        elif dec2.next == "revise_or_stop":
            trace.final_status = "needs_insertion_revision"
        else:
            trace.final_status = "failed"
        write_trace(case_output / "agent_trace.json", trace)

        if use_llm_decision:
            self._append_llm_decision_step(
                trace=trace,
                case=case,
                case_output=case_output,
                latest_verification=ver2,
                allowed_tools=llm_allowed_tools or DEFAULT_LLM_ALLOWED_TOOLS,
                execute_tool=execute_llm_tool,
                dry_run=dry_run,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
            )
            write_trace(case_output / "agent_trace.json", trace)
        return trace

    def _append_llm_decision_step(
        self,
        trace: AgentTrace,
        case: CaseConfig,
        case_output: Path,
        latest_verification: Verification,
        allowed_tools: list[str],
        execute_tool: bool,
        dry_run: bool,
        llm_base_url: str,
        llm_model: str,
    ) -> None:
        load_project_env(self.project_root)
        try:
            client = OpenAICompatibleClient(base_url=llm_base_url, model=llm_model)
            llm_decision = choose_next_action(client, trace, latest_verification, allowed_tools)
            observation = Observation(
                status="success",
                summary=f"LLM selected {llm_decision.get('next')}: {llm_decision.get('reason')}",
                data={"llm_decision": llm_decision, "allowed_tools": allowed_tools, "execute_tool": execute_tool},
            )
            verification = Verification(stage="llm_decision", status="pass", score=1.0)
            decision = Decision(
                next=str(llm_decision.get("next", "stop")),
                reason=str(llm_decision.get("reason", "")),
                revision={"params": llm_decision.get("params", {})},
            )
        except Exception as exc:
            observation = Observation(status="failed", summary=f"LLM decision failed: {exc!r}")
            verification = Verification(stage="llm_decision", status="fail", score=0.0, problems=[observation.summary])
            decision = Decision(next="stop", reason="LLM decision failed.")

        step_index = len(trace.steps) + 1
        trace.steps.append(TraceStep(
            index=step_index,
            thought="Ask the OpenAI-compatible LLM policy to choose the next whitelisted tool or stop.",
            action=ToolAction(tool="llm.choose_next_action", params={"allowed_tools": allowed_tools}),
            observation=observation,
            verification=verification,
            decision=decision,
        ))

        selected_tool = decision.next
        if not execute_tool or selected_tool == "stop" or observation.status != "success":
            trace.final_status = f"llm_decision_{selected_tool}"
            return

        params = observation.data.get("llm_decision", {}).get("params", {})
        if not isinstance(params, dict):
            params = {}
        tool_obs = self.tools.call(selected_tool, case, output_dir=case_output, dry_run=dry_run, **params)
        tool_ver = Verification(
            stage="llm_selected_tool",
            status="pass" if tool_obs.status in {"success", "skipped"} else "fail",
            score=1.0 if tool_obs.status == "success" else 0.0,
            problems=[] if tool_obs.status in {"success", "skipped"} else [tool_obs.summary],
        )
        tool_dec = Decision(next="stop", reason="Executed LLM-selected tool once.")
        trace.steps.append(TraceStep(
            index=len(trace.steps) + 1,
            thought="Execute the LLM-selected whitelisted tool once.",
            action=ToolAction(tool=selected_tool, params=params),
            observation=tool_obs,
            verification=tool_ver,
            decision=tool_dec,
        ))
        trace.final_status = "llm_tool_executed" if tool_ver.status == "pass" else "llm_tool_failed"


def load_case(config_path: Path, case_id: str | None, project_root: Path) -> CaseConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(f"no cases in {config_path}")
    raw = None
    if case_id is None:
        raw = cases[0]
    else:
        for item in cases:
            if item.get("case_id") == case_id:
                raw = item
                break
    if raw is None:
        raise ValueError(f"case_id not found: {case_id}")
    return CaseConfig.from_dict(raw, base_dir=project_root)
