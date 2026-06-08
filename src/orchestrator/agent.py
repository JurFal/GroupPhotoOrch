"""Rule-based ReAct orchestrator for the current Computer-Graphics tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .env import load_project_env
from .llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleClient
from .policies.llm_decision_policy import choose_next_action
from .policies.revision_policy import build_revision_params, decide_after_verification, revision_limit_for
from .schemas import AgentTrace, CaseConfig, Decision, Observation, ToolAction, TraceStep, Verification
from .trace import write_trace
from .tools.toolbox import ToolBox
from .verifiers.insertion_verifier import verify_candidate_summaries
from .verifiers.lighting_verifier import choose_mrf_params, verify_lighting_plan, verify_mrf_composite
from .verifiers.tone_verifier import verify_hsv_tone_alignment
from .verifiers.vision_verifier import verify_existing_artifacts


DEFAULT_LLM_ALLOWED_TOOLS = [
    "compositing.align_tone_hsv",
    "compositing.run_light_smoke",
    "compositing.compose_top_candidate",
    "compositing.compose_all_candidates",
    "vision.extract_metadata_from_masks",
    "vision.inspect_existing_artifacts",
]


class ReActOrchestrator:
    def __init__(self, project_root: Path, output_root: Path | None = None) -> None:
        self.project_root = project_root
        self.output_root = output_root or project_root / "orchestrator" / "outputs"
        self.tools = ToolBox()

    def _append_step(
        self,
        trace: AgentTrace,
        *,
        thought: str,
        tool: str,
        params: dict[str, Any],
        observation: Observation,
        verification: Verification,
        decision: Decision,
    ) -> None:
        trace.steps.append(TraceStep(
            index=len(trace.steps) + 1,
            thought=thought,
            action=ToolAction(tool=tool, params=params),
            observation=observation,
            verification=verification,
            decision=decision,
        ))

    def _run_verified_tool(
        self,
        trace: AgentTrace,
        case: CaseConfig,
        case_output: Path,
        *,
        stage: str,
        tool: str,
        params: dict[str, Any],
        thought: str,
        retry_thought: str,
        verify_fn: Any,
        pass_next: str,
        dry_run: bool,
    ) -> tuple[Observation, Verification, Decision, dict[str, Any]]:
        """Run a tool, verify it, and retry locally with revised params when safe."""
        limit = revision_limit_for(stage, case.revision_limits)
        current_params = dict(params)
        attempt = 0

        while True:
            obs = self.tools.call(tool, case, output_dir=case_output, dry_run=dry_run, **current_params)
            ver = verify_fn(obs)
            dec = decide_after_verification(stage, ver, pass_next=pass_next)
            self._append_step(
                trace,
                thought=thought if attempt == 0 else retry_thought,
                tool=tool,
                params=current_params,
                observation=obs,
                verification=ver,
                decision=dec,
            )

            should_retry = ver.status in {"needs_revision", "fail"} and attempt < limit and not dry_run
            if not should_retry:
                return obs, ver, dec, current_params

            revised_params = build_revision_params(stage, tool, current_params, ver, attempt + 1)
            if revised_params is None or revised_params == current_params:
                return obs, ver, dec, current_params

            attempt += 1
            dec.next = tool
            dec.reason = (
                f"{stage} verifier was not good enough; retrying locally with revised params "
                f"(attempt {attempt}/{limit})."
            )
            dec.revision = {"params": revised_params, "problems": ver.problems}
            current_params = revised_params

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
        obs2, ver2, dec2, _ = self._run_verified_tool(
            trace,
            case,
            case_output,
            stage="insertion",
            tool="geometry.find_insertion_candidates",
            params={"top_k": case.top_k},
            thought="Use the current PersonInserter interface to generate candidate patches; serialize only lightweight summaries.",
            retry_thought="Insertion verification was not good enough; relax candidate search and rerun only stage 2.",
            verify_fn=verify_candidate_summaries,
            pass_next="align_tone",
            dry_run=dry_run,
        )

        latest_verification = ver2
        if dec2.next == "align_tone":
            obs3, ver3, dec3, _ = self._run_verified_tool(
                trace,
                case,
                case_output,
                stage="tone_alignment",
                tool="compositing.align_tone_hsv",
                params={
                    "max_hue_shift_deg": 18.0,
                    "min_saturation_ratio": 0.75,
                    "max_saturation_ratio": 1.35,
                    "value_strength": 0.0,
                    "strength": 1.0,
                    "min_valid_pixels": 128,
                },
                thought="Align the inserted person's HSV tone against masked people in the group photo before final MRF compositing.",
                retry_thought="Tone verification was not good enough; adjust HSV limits and rerun only tone alignment.",
                verify_fn=verify_hsv_tone_alignment,
                pass_next="plan_mrf_compositing",
                dry_run=dry_run,
            )
            latest_verification = ver3
            if dec3.next == "plan_mrf_compositing":
                ver4 = verify_lighting_plan(obs3, obs2)
                mrf_params = choose_mrf_params(obs3, obs2)
                mrf_params.pop("candidate_rank", None)
                mrf_params["candidate_count"] = int(obs2.data.get("candidate_count", case.top_k))
                dec4 = decide_after_verification("lighting_plan", ver4, pass_next="compose_all_candidates")
                trace.steps.append(TraceStep(
                    index=4,
                    thought="HSV fixes color temperature/tone, but MRF is still needed to smooth local illumination into the group lighting.",
                    action=ToolAction(tool="lighting.verify_and_budget_mrf", params={"case_id": case.case_id}),
                    observation=Observation(
                        status="success",
                        summary=f"Lighting verifier selected MRF budget: max_iter={mrf_params['max_iter']}, max_crop_size={mrf_params['max_crop_size']}.",
                        data={"mrf_params": mrf_params},
                    ),
                    verification=ver4,
                    decision=dec4,
                ))
                latest_verification = ver4

                obs5, ver5, dec5, mrf_params = self._run_verified_tool(
                    trace,
                    case,
                    case_output,
                    stage="mrf_compositing",
                    tool="compositing.compose_all_candidates",
                    params=mrf_params,
                    thought="Execute final ImageCompositor MRF pass for every stage-2 candidate so downstream review sees all options.",
                    retry_thought="MRF composite verification was not good enough; adjust compose budget and rerun only stage 3.",
                    verify_fn=verify_mrf_composite,
                    pass_next="final_composite_ready",
                    dry_run=dry_run,
                )
                latest_verification = ver5
                if dec5.next == "final_composite_ready":
                    trace.final_status = "success_dry_run" if dry_run else "all_candidates_composed"
                elif dec5.next == "revise_or_stop":
                    trace.final_status = "needs_mrf_revision"
                else:
                    trace.final_status = "failed"
            elif dec3.next == "revise_or_stop":
                trace.final_status = "needs_tone_alignment_revision"
            else:
                trace.final_status = "failed"
        elif dec2.next == "ready_for_optional_compositing":
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
                latest_verification=latest_verification,
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
        if observation.status != "success":
            observation.data["preserved_final_status"] = trace.final_status
            decision.reason = (
                f"{decision.reason} LLM decision failed after the deterministic workflow; "
                f"preserving workflow final_status={trace.final_status}."
            ).strip()
            return
        if selected_tool == "stop" or not execute_tool:
            observation.data["advisory_only"] = True
            observation.data["preserved_final_status"] = trace.final_status
            decision.reason = (
                f"{decision.reason} Advisory LLM decision only; preserving workflow final_status={trace.final_status}."
            ).strip()
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
