"""OpenAI-compatible chat client for the ReAct orchestrator."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "nex-agi/Nex-N2-Pro"


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    content: str
    raw: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError("Missing API key. Set API_KEY or SILICONFLOW_API_KEY in .env or environment.")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        normalized = [m.__dict__ if isinstance(m, ChatMessage) else m for m in messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.chat_completions_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        raw = json.loads(raw_text)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {raw}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        return ChatResponse(content=content, raw=raw)
