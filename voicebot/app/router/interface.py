"""Router contract: every turn the router decides what the caller is doing.

Decisions: continue | faq | lookup | switch | resume | unclear | human | end
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

DECISIONS = ["continue", "faq", "lookup", "switch", "resume", "unclear", "human", "end"]


@dataclass
class RouteContext:
    utterance: str
    text: str                                   # cleaned (lowercase, variants unified)
    active: str | None = None                   # active workflow name
    step: str | None = None
    expects: str = "text"                       # yes_no | choice | number | date | text
    pending_kind: str | None = None             # run-level pending
    session_pending: str | None = None          # switch | resume | lookup_choose | faq_confirm ...
    suspended: list[str] = field(default_factory=list)
    last_bot: str = ""
    workflows: dict[str, dict] = field(default_factory=dict)    # name -> manifest
    can_interrupt: bool = True
    has_progress: bool = False
    mentions_test: bool = False                 # utterance resolves to a test name
    slot_answer: bool = False                   # utterance parses as the answer to the current question
    faq_status: str = "none"                    # none | confirm | answer
    faq_score: float = 0.0


@dataclass
class RouteDecision:
    decision: str
    workflow: str | None = None
    confidence: float = 1.0
    source: str = "rules"
    reason: str = ""
    tried: list[dict] = field(default_factory=list)   # [{source, decision, confidence, note}] for the UI

    def as_dict(self) -> dict:
        return {"decision": self.decision, "workflow": self.workflow, "confidence": round(self.confidence, 3),
                "source": self.source, "reason": self.reason, "tried": self.tried}


class Router(Protocol):
    name: str

    async def route(self, ctx: RouteContext) -> RouteDecision | None:
        """Return a decision, or None when unavailable / not applicable."""
