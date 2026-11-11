"""Thin async client for an OpenAI-compatible endpoint (vLLM / llama.cpp) serving Qwen3-8B.

Everything degrades gracefully: with no LLM_BASE_URL (or on error/timeout) calls return None and
the bot continues with its rule-based paths.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from app.config import Settings, get_settings

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_loose(text: str) -> dict | None:
    """Parse a JSON object from model output (tolerates code fences and <think> blocks)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class LLM:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._client = None
        self.last_error: str = ""
        self.last_ms: float = 0.0
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.s.llm_base_url)

    def _get(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.s.llm_base_url, api_key=self.s.llm_api_key, timeout=20.0)
        return self._client

    async def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int | None = None,
                   temperature: float | None = None, timeout: float = 15.0) -> str | None:
        if not self.enabled:
            return None
        kwargs: dict[str, Any] = dict(
            model=self.s.llm_model,
            messages=messages,
            temperature=self.s.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or max(self.s.llm_max_tokens, 200),
            extra_body={"chat_template_kwargs": {"enable_thinking": self.s.llm_enable_thinking}},
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        self.calls += 1
        try:
            import asyncio
            resp = await asyncio.wait_for(self._get().chat.completions.create(**kwargs), timeout=timeout)
            self.last_error = ""
            return resp.choices[0].message.content or ""
        except Exception as e:  # network, timeout, bad request
            if json_mode and "response_format" in str(e).lower():
                kwargs.pop("response_format", None)
                try:
                    resp = await self._get().chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                except Exception as e2:
                    e = e2
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            return None
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000

    async def chat_json(self, messages: list[dict], **kw) -> dict | None:
        out = await self.chat(messages, json_mode=True, **kw)
        return parse_json_loose(out) if out else None


class FakeLLM(LLM):
    """Scripted LLM for tests: handler(messages) -> str | dict | None."""

    def __init__(self, handler):
        super().__init__(Settings(llm_base_url="fake://"))
        self.handler = handler
        self.prompts: list[list[dict]] = []

    async def chat(self, messages, **kw):
        self.prompts.append(messages)
        self.calls += 1
        out = self.handler(messages)
        if isinstance(out, dict):
            return json.dumps(out)
        return out
