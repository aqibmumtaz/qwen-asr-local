"""Laya (non-autoregressive "System 1" decision model) as the primary router.

One forward pass answers typed questions over the state (utterance, last bot question, active workflow, what the
bot is waiting for): ~150-300 ms on CPU. Zero-shot quality on Roman Urdu is limited, so the chain only trusts it above
LAYA_CONFIDENCE_MIN and otherwise falls back to Qwen, then rules. Fine-tuning/calibration on logged routing
decisions is the improvement path (scripts/run_router_bakeoff.py measures it).
"""
from __future__ import annotations

import asyncio
import threading

from app.config import Settings
from app.router.interface import DECISIONS, RouteContext, RouteDecision

_ALIASES = {"laya-multilingual": "multilingual", "laya": "english", "laya-typed-decisions": "typed-decisions"}

INTENT_CRITERIA = {
    "continue": "the caller is answering the assistant's last question or giving details (a name, age, address, date, "
                "phone number, yes or no, a choice) for the active task",
    "faq": "the caller asks a general question about opening hours, charges, payment, report time or the helpline",
    "lookup": "the caller asks the price or details of a specific lab test",
    "switch": "the caller asks for a different task than the active one, such as checking, cancelling or "
              "rescheduling a booking, or making a new booking",
    "resume": "the caller wants to go back to a task that was paused",
    "unclear": "the caller says something unrelated or the meaning cannot be understood",
    "human": "the caller asks to speak to a person, an agent or an operator",
    "end": "the caller says goodbye or says they are finished",
}


class LayaDecisionRouter:
    name = "laya"

    def __init__(self, settings: Settings):
        self.s = settings
        self._agent = None
        self._lock = threading.Lock()
        self.load_error = ""

    def _load(self):
        with self._lock:
            if self._agent is not None:
                return self._agent
            import torch
            # Python 3.12 + torch 2.2 (Intel Mac) cannot use torch.compile; make it a no-op before transformers loads.
            if not getattr(torch, "_voicebot_compile_patched", False):
                torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
                torch._voicebot_compile_patched = True
            import laya
            model = _ALIASES.get(self.s.laya_model, self.s.laya_model)
            dev = None if self.s.laya_device == "auto" else self.s.laya_device
            self._agent = laya.load(model, device=dev or "cpu")
            return self._agent

    def warm(self) -> None:
        self._load()

    def questions(self, ctx: RouteContext) -> dict:
        qs = {"intent": {"type": "choice", "instructions": "What is the caller doing with `utterance`?",
                         "criteria": dict(INTENT_CRITERIA)}}
        if len(ctx.workflows) > 1:
            crit = {n: m.get("description", n) for n, m in ctx.workflows.items()}
            qs["workflow"] = {"type": "choice", "instructions": "Which task does `utterance` ask for?", "criteria": crit}
        return qs

    def state(self, ctx: RouteContext) -> dict:
        return {"utterance": ctx.utterance, "assistant_last_question": ctx.last_bot[-200:],
                "active_task": ctx.active or "none", "assistant_is_waiting_for": ctx.expects}

    def _predict(self, ctx: RouteContext) -> RouteDecision | None:
        agent = self._load()
        res = agent.predict(self.state(ctx), self.questions(ctx))
        a = res["answers"]["intent"]
        dec = a["choice"]
        if dec not in DECISIONS:
            return None
        conf = float(a.get("confidence", a.get("answer_confidence", 0.0)))
        wf = None
        if dec == "switch":
            w = res["answers"].get("workflow")
            wf = w["choice"] if w else next(iter(ctx.workflows), None)
            if wf not in ctx.workflows:
                return None
            conf = min(conf, float(w.get("confidence", conf))) if w else conf
        return RouteDecision(dec, wf, conf, source="laya", reason="laya")

    async def route(self, ctx: RouteContext) -> RouteDecision | None:
        try:
            return await asyncio.to_thread(self._predict, ctx)
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"[:200]
            raise
