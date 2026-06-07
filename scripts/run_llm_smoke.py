#!/usr/bin/env python3
"""Smoke test the OpenAI-compatible LLM client."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "orchestrator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchestrator.env import load_project_env  # noqa: E402
from orchestrator.llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Return JSON with one key named status and value ok.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="Validate env/config without making a network request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env(ROOT)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "base_url": args.base_url,
            "model": args.model,
            "env_loaded": True,
        }, ensure_ascii=False, indent=2))
        return

    client = OpenAICompatibleClient(base_url=args.base_url, model=args.model)
    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a concise assistant. Return only what the user asks for."},
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.1,
        max_tokens=args.max_tokens,
    )
    print(json.dumps({
        "status": "success",
        "base_url": args.base_url,
        "model": args.model,
        "content": response.content,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
