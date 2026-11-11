"""Shared dataclasses for sessions, workflow runs and turn results."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable


@dataclass
class Pending:
    """A question the bot asked that the next utterance is expected to answer."""
    kind: str                    # choose | confirm_entity | spell | confirm_slot | confirm_step | which_slot |
    #                              switch | resume | lookup_choose | lookup_confirm | faq_confirm
    data: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""             # text the bot said (re-asked after an interruption)


@dataclass
class WorkflowRun:
    name: str
    slots: dict[str, Any] = field(default_factory=dict)
    step: str | None = None
    retries: dict[str, int] = field(default_factory=dict)
    pending: Pending | None = None
    scratch: dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    status: str = "active"       # active | suspended | done | cancelled
    last_ask: str = ""

    def snapshot(self) -> dict:
        return {"workflow": self.name, "step": self.step, "status": self.status, "slots": self.slots,
                "pending": self.pending.kind if self.pending else None}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    facts: dict[str, Any] = field(default_factory=dict)        # shared: patient/caller name, phone, caller_phone
    active: WorkflowRun | None = None
    suspended: list[WorkflowRun] = field(default_factory=list)
    pending: Pending | None = None                              # session-level (switch / resume / lookup)
    history: list[dict] = field(default_factory=list)
    turn: int = 0
    ended: bool = False
    transfer: str | None = None
    last_bot: str = ""
    greeted: bool = False
    unclear_count: int = 0
    lookup: Any = None                                         # temporary run used by lookup sub-dialogue
    scratch: dict = field(default_factory=dict)


@dataclass
class TurnCtx:
    kit: Any
    llm: Any
    backend: Any
    faq: Any
    settings: Any
    today: date
    persona: dict[str, str]
    emit: Callable[[str, dict], None] = lambda kind, payload: None


@dataclass
class TurnOut:
    replies: list[str] = field(default_factory=list)
    done: bool = False           # workflow finished
    transfer: str | None = None  # reason for human handoff
    end_call: bool = False

    def say(self, *texts: str) -> "TurnOut":
        self.replies.extend(t for t in texts if t)
        return self


class SafeDict(dict):
    def __missing__(self, key):  # leave unknown placeholders visible for debugging
        return "{" + key + "}"


def fmt(template: str, **vals: Any) -> str:
    return template.format_map(SafeDict(**vals))
