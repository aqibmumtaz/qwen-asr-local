"""Qwen3-8B router: constrained JSON decision with a confidence value."""
from __future__ import annotations

import json

from app.llm.client import LLM
from app.router.interface import DECISIONS, RouteContext, RouteDecision

SYSTEM = """You are the router of a Pakistani diagnostic lab's phone assistant. The caller speaks Urdu; text is
noisy Roman Urdu from speech recognition (spelling varies, words may be misheard).
Decide what the caller's latest utterance is doing. Answer with JSON only:
{"decision": "<one of: continue, faq, lookup, switch, resume, unclear, human, end>", "workflow": "<name or null>", "confidence": <0..1>}

Decisions:
- continue: the utterance answers the assistant's last question or gives details for the active task. THIS IS THE DEFAULT when a task is active.
- faq: a general question answered by a fixed FAQ (timings, charges, payment, report time, helpline).
- lookup: asks the price or details of a specific lab test.
- switch: clearly asks for a DIFFERENT task than the active one (or starts a task when none is active). Set "workflow".
- resume: wants to go back to a task that was paused.
- unclear: you cannot tell what the caller wants.
- human: asks to speak to a person.
- end: says goodbye / is done.
Never choose switch because of a single ambiguous word. If the utterance could be an answer to the last question, choose continue."""


class QwenRouter:
    name = "qwen"

    def __init__(self, llm: LLM, min_conf: float = 0.7):
        self.llm = llm
        self.min_conf = min_conf

    def _user(self, ctx: RouteContext) -> str:
        wfs = {n: {"description": m.get("description", ""), "examples": m.get("examples", [])[:6]}
               for n, m in ctx.workflows.items()}
        return json.dumps({
            "utterance": ctx.utterance,
            "assistant_last_question": ctx.last_bot[-300:],
            "active_workflow": ctx.active,
            "expecting": ctx.expects,
            "paused_workflows": ctx.suspended,
            "workflows": wfs,
        }, ensure_ascii=False)

    async def route(self, ctx: RouteContext) -> RouteDecision | None:
        if not self.llm or not self.llm.enabled:
            return None
        j = await self.llm.chat_json([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": self._user(ctx)}], max_tokens=80, temperature=0.0)
        if not j:
            return None
        dec = str(j.get("decision", "")).strip().lower()
        if dec not in DECISIONS:
            return None
        wf = j.get("workflow")
        if wf in (None, "", "null") or wf not in ctx.workflows:
            wf = None
        if dec == "switch" and wf is None:
            return None
        try:
            conf = float(j.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        return RouteDecision(dec, wf, conf, source="qwen", reason="llm")
