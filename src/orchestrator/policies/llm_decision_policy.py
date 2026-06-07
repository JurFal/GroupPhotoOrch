"""LLM-assisted decision helper for choosing among registered tools."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..llm_client import OpenAICompatibleClient
from ..schemas import AgentTrace, Verification

SYSTEM_PROMPT = """You are a constrained ReAct decision policy for a computer graphics image-insertion agent.
Choose only from the provided tool names or choose stop. Do not invent tools.
Prefer safe, cheap diagnostic tools unless the user explicitly asks to execute expensive tools.
Return compact JSON only.
"""


def choose_next_action(
    client: OpenAICompatibleClient,
    trace: AgentTrace,
    latest_verification: Verification,
    allowed_tools: list[str],
) -> dict[str, Any]:
    compact_steps = []
    for step in trace.steps[-4:]:
        compact_steps.append({
            "index": step.index,
            "tool": step.action.tool,
            "observation_summary": step.observation.summary,
            "verification": asdict(step.verification),
            "decision": asdict(step.decision),
        })
    payload = {
        "case_id": trace.case_id,
        "goal": trace.goal,
        "final_status": trace.final_status,
        "recent_steps": compact_steps,
        "latest_verification": asdict(latest_verification),
        "allowed_tools": allowed_tools,
        "instruction": (
            "Return JSON: {\"next\": tool_name_or_stop, \"reason\": string, \"params\": object}. "
            "If the workflow already has usable candidates and no further action is needed, choose stop."
        ),
    }
    response = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        decision = {"next": "stop", "reason": "LLM did not return valid JSON", "raw": response.content, "params": {}}
    if decision.get("next") not in allowed_tools and decision.get("next") != "stop":
        decision = {
            "next": "stop",
            "reason": f"LLM selected disallowed tool: {decision.get('next')}",
            "params": {},
        }
    decision.setdefault("params", {})
    return decision
