#!/usr/bin/env python3
"""Run the P0 ReAct orchestrator on a configured demo case."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "orchestrator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestrator.agent import ReActOrchestrator, load_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "orchestrator" / "configs" / "demo_cases.json")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-llm-decision", action="store_true", help="Ask the configured LLM to choose the next whitelisted tool after default steps.")
    parser.add_argument("--execute-llm-tool", action="store_true", help="Execute the LLM-selected tool once. By default only records the decision.")
    parser.add_argument("--llm-allowed-tools", default="", help="Comma-separated whitelist override for LLM decisions.")
    parser.add_argument("--llm-base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--llm-model", default="nex-agi/Nex-N2-Pro")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    case = load_case(config_path, args.case_id, project_root=ROOT)
    agent = ReActOrchestrator(project_root=ROOT)
    allowed_tools = [x.strip() for x in args.llm_allowed_tools.split(",") if x.strip()] or None
    trace = agent.run_case(
        case,
        dry_run=args.dry_run,
        use_llm_decision=args.use_llm_decision,
        execute_llm_tool=args.execute_llm_tool,
        llm_allowed_tools=allowed_tools,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
    )
    out = ROOT / "orchestrator" / "outputs" / case.case_id / "agent_trace.json"
    print(f"case={trace.case_id} final_status={trace.final_status}")
    print(f"trace={out}")
    for step in trace.steps:
        print(f"#{step.index} {step.action.tool}: {step.observation.summary} -> {step.verification.status}")


if __name__ == "__main__":
    main()
