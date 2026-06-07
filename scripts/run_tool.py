#!/usr/bin/env python3
"""Call a single orchestrator tool by registry name."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "orchestrator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestrator.agent import load_case  # noqa: E402
from orchestrator.tools.toolbox import ToolBox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", help="Tool name, e.g. vision.generate_sam3_masks")
    parser.add_argument("--config", type=Path, default=ROOT / "orchestrator" / "configs" / "demo_cases.json")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--params-json",
        default="{}",
        help="JSON object with tool-specific params.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise SystemExit("--params-json must decode to an object")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    case = load_case(config_path, args.case_id, project_root=ROOT)
    output_dir = ROOT / "orchestrator" / "outputs" / case.case_id
    obs = ToolBox().call(args.tool, case, output_dir=output_dir, dry_run=args.dry_run, **params)
    print(json.dumps(obs.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
