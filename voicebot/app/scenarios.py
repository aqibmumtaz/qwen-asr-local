"""Scenario files (eval/scenarios/*.yaml): scripted caller lines with expectations.

The same files drive the automated eval, the UI scenario picker (guided / auto-play) and `check_workflow`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from app.backend.mock_api import MockBackend
from app.bot import Bot
from app.config import ROOT, Settings, get_settings

SCENARIO_DIR = ROOT / "eval" / "scenarios"


def load_scenarios(d: Path = SCENARIO_DIR) -> list[dict]:
    out: list[dict] = []
    for p in sorted(d.glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
        for sc in (data if isinstance(data, list) else [data]):
            sc["_file"] = p.name
            out.append(sc)
    return out


@dataclass
class StepResult:
    say: str
    reply: str
    route: dict
    ok: bool
    failures: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    id: str
    title: str
    ok: bool
    steps: list[StepResult]
    failures: list[str]
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "ok": self.ok, "tags": self.tags, "failures": self.failures,
                "steps": [s.__dict__ for s in self.steps]}


def _subset(expected: dict, actual: dict) -> list[str]:
    bad = []
    for k, v in expected.items():
        a = actual.get(k)
        if isinstance(v, dict) and isinstance(a, dict):
            bad += [f"{k}.{x}" for x in _subset(v, a)]
        elif isinstance(v, list) and isinstance(a, list):
            names = [i["name"] if isinstance(i, dict) and "name" in i else i for i in a]
            if sorted(map(str, names)) != sorted(map(str, v)):
                bad.append(f"{k}: expected {v}, got {names}")
        elif a != v:
            bad.append(f"{k}: expected {v!r}, got {a!r}")
    return bad


def make_bot(settings: Settings | None = None, routers: dict | None = None, llm=None) -> Bot:
    return Bot(settings=settings or get_settings(), llm=llm, routers=routers if routers is not None else {})


def check_expect(exp: dict, r, session, tools_this_turn: list[str]) -> list[str]:
    """Compare one turn's result with a step's `expect` block; returns a list of failure messages."""
    bad: list[str] = []
    if "router" in exp and r.route["decision"] != exp["router"]:
        bad.append(f"router: expected {exp['router']}, got {r.route['decision']}")
    if "workflow" in exp:
        act = r.state["active"]["workflow"] if r.state["active"] else None
        if act != exp["workflow"]:
            bad.append(f"active workflow: expected {exp['workflow']}, got {act}")
    if "step" in exp:
        act_step = r.state["active"]["step"] if r.state["active"] else None
        if act_step != exp["step"]:
            bad.append(f"step: expected {exp['step']}, got {act_step}")
    if "slots" in exp:
        bad += _subset(exp["slots"], (r.state["active"] or {}).get("slots", {}))
    for sub in exp.get("bot_contains", []):
        if str(sub).lower() not in r.text.lower():
            bad.append(f"reply missing '{sub}'")
    for sub in exp.get("bot_not_contains", []):
        if str(sub).lower() in r.text.lower():
            bad.append(f"reply should not contain '{sub}'")
    if "pending" in exp:
        p = r.state["pending"] or ((r.state["active"] or {}).get("pending"))
        if p != exp["pending"]:
            bad.append(f"pending: expected {exp['pending']}, got {p}")
    if "transfer" in exp and r.transfer != exp["transfer"]:
        bad.append(f"transfer: expected {exp['transfer']}, got {r.transfer}")
    if "ended" in exp and r.ended != exp["ended"]:
        bad.append(f"ended: expected {exp['ended']}, got {r.ended}")
    if "suspended" in exp and len(r.state["suspended"]) != exp["suspended"]:
        bad.append(f"suspended: expected {exp['suspended']}, got {len(r.state['suspended'])}")
    if "tool" in exp and exp["tool"] not in tools_this_turn:
        bad.append(f"tool {exp['tool']} not called this turn")
    return bad


def check_final(fin: dict, backend, seen_tools: list[str]) -> list[str]:
    failures: list[str] = []
    created = [b for b in backend.bookings.values() if b["ref"] != "HS-1000"]
    if "booking" in fin:
        if not created:
            failures.append("final: no booking was created")
        else:
            failures += [f"final booking {x}" for x in _subset(fin["booking"], created[-1])]
    if fin.get("no_booking") and created:
        failures.append("final: a booking was created but should not have been")
    for ref, status in (fin.get("booking_status") or {}).items():
        got = backend.bookings.get(ref, {}).get("status")
        if got != status:
            failures.append(f"final: booking {ref} status expected {status}, got {got}")
    for ref, fields in (fin.get("booking_fields") or {}).items():
        failures += [f"final {ref}.{x}" for x in _subset(fields, backend.bookings.get(ref, {}))]
    for t in fin.get("tools_called", []):
        if t not in seen_tools:
            failures.append(f"final: tool {t} never called")
    return failures


def new_scenario_session(bot: Bot, sc: dict):
    today = date.fromisoformat(str(sc.get("today", "2026-10-21")))
    return bot.new_session(sc.get("caller_phone"), today=today, setup=sc.get("setup"))


async def run_scenario(sc: dict, bot: Bot | None = None, settings: Settings | None = None,
                       routers: dict | None = None, llm=None) -> ScenarioResult:
    bot = bot or make_bot(settings, routers, llm)
    session = new_scenario_session(bot, sc)
    backend = session.scratch["backend"]
    if sc.get("greet", False):
        bot.greeting(session)
    steps: list[StepResult] = []
    failures: list[str] = []
    seen_tools: list[str] = []
    for i, st in enumerate(sc["steps"]):
        if st.get("fail_backend"):
            backend.fail_next = st["fail_backend"]
        r = await bot.turn(session, st["say"])
        tools = [e["name"] for e in r.events if e["kind"] == "tool"]
        seen_tools += tools
        bad = check_expect(st.get("expect") or {}, r, session, tools)
        steps.append(StepResult(st["say"], r.text, r.route, not bad, bad, r.state))
        failures += [f"step {i + 1} ({st['say']!r}): {b}" for b in bad]
        if session.ended and i < len(sc["steps"]) - 1 and not st.get("continue_after_end"):
            break
    failures += check_final(sc.get("final") or {}, backend, seen_tools)
    return ScenarioResult(sc["id"], sc.get("title", sc["id"]), not failures, steps, failures, sc.get("tags", []))


async def run_many(scs: list[dict], **kw) -> list[ScenarioResult]:
    return [await run_scenario(copy.deepcopy(sc), **kw) for sc in scs]
