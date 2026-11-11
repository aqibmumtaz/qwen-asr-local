"""The bot: one object that turns a caller utterance into replies.

Per turn:  clean text -> route (what is the caller doing?) -> dispatch (workflow / FAQ / lookup / switch /
resume / human / end) -> reply + state snapshot + events for the UI and logs.
It is framework-free: LiveKit, the chat UI and the scenario runner all call `Bot.turn`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from app.backend.mock_api import MockBackend
from app.config import ROOT, Settings, get_settings
from app.datakit.resolve import DataKit
from app.faq import FaqBook
from app.flow import slots as sl
from app.flow.form import Workflow
from app.flow.types import Pending, Session, TurnCtx, TurnOut, WorkflowRun, fmt
from app.llm.client import LLM
from app.nlp import normalizers as nz
from app.persona import persona_vars
from app.router.chain import ChainRouter
from app.router.interface import RouteContext, RouteDecision
from app.router.llm_router import QwenRouter


@dataclass
class TurnResult:
    replies: list[str]
    text: str
    route: dict
    events: list[dict]
    state: dict
    timings: dict
    ended: bool = False
    transfer: str | None = None


class Bot:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None, backend=None, kit: DataKit | None = None,
                 faq: FaqBook | None = None, routers: dict | None = None, today: date | None = None,
                 backend_factory=None):
        self.s = settings or get_settings()
        self.llm = llm or LLM(self.s)
        # one backend per session by default (mock bookings are isolated); pass `backend` to share one
        self.backend_factory = backend_factory or ((lambda: backend) if backend is not None else MockBackend)
        self.kit = kit or DataKit(settings=self.s)
        self.faq = faq or FaqBook(settings=self.s)
        self.persona = persona_vars(self.s)
        self.today_override = today
        self.workflows: dict[str, Workflow] = {}
        for d in sorted((ROOT / "workflows").iterdir()):
            if (d / "workflow.yaml").exists():
                wf = Workflow(d)
                self.workflows[wf.name] = wf
        self.manifests = {n: w.manifest for n, w in self.workflows.items()}
        rs = routers if routers is not None else self._default_routers()
        self.router = ChainRouter(self.s, rs)
        self.sessions: dict[str, Session] = {}

    def _default_routers(self) -> dict:
        rs: dict = {"qwen": QwenRouter(self.llm, self.s.qwen_router_conf_min)}
        try:
            from app.router.laya_router import LayaDecisionRouter
            rs["laya"] = LayaDecisionRouter(self.s)
        except Exception:
            pass
        return rs

    # ── sessions ──
    def new_session(self, caller_phone: str | None = None, today: date | None = None, setup: dict | None = None) -> Session:
        s = Session()
        s.scratch["backend"] = self.backend_factory()
        s.scratch["today"] = today
        if hasattr(s.scratch["backend"], "set_today"):
            s.scratch["backend"].set_today(today or self.today())
        for d, part in ((setup or {}).get("blocked_slots") or []):
            if hasattr(s.scratch["backend"], "blocked_slots"):
                s.scratch["backend"].blocked_slots.add((d, part))
        if caller_phone:
            r = nz.normalize_phone(caller_phone)
            s.facts["caller_phone"] = r.digits if r.valid else caller_phone
        self.sessions[s.id] = s
        return s

    def today(self) -> date:
        return self.today_override or datetime.now().date()

    def _ctx(self, events: list[dict], session: Session) -> TurnCtx:
        return TurnCtx(kit=self.kit, llm=self.llm, backend=session.scratch["backend"], faq=self.faq, settings=self.s,
                       today=session.scratch.get("today") or self.today(), persona=self.persona,
                       emit=lambda kind, payload: events.append({"kind": kind, **payload}))

    # ── text helpers ──
    def _p(self, template: str, **kw) -> str:
        return fmt(template, **self.persona, **kw)

    def greeting(self, session: Session) -> TurnResult:
        session.greeted = True
        txt = self._p("Assalam o alaikum, Chughtai Lab mein khush aamdeed. Main {bot_name} hoon. "
                      "Aap home sampling booking karna chahte hain, ya kisi booking ka status dekhna chahte hain?")
        session.last_bot = txt
        self._log(session, "", txt, {}, [])
        return TurnResult([txt], txt, {}, [], self.snapshot(session), {})

    def snapshot(self, session: Session) -> dict:
        return {
            "session": session.id, "ended": session.ended, "transfer": session.transfer,
            "active": session.active.snapshot() if session.active else None,
            "suspended": [r.snapshot() for r in session.suspended],
            "pending": session.pending.kind if session.pending else None,
            "facts": {k: v for k, v in session.facts.items()},
        }

    # ── main entry ──
    async def turn(self, session: Session, utterance: str) -> TurnResult:
        t0 = time.perf_counter()
        events: list[dict] = []
        ctx = self._ctx(events, session)
        session.turn += 1
        clean = nz.clean_text(utterance)
        timings: dict[str, float] = {}

        rc = self._route_context(ctx, session, utterance, clean)
        t1 = time.perf_counter()
        decision = await self.router.route(rc)
        timings["route_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        events.append({"kind": "route", **decision.as_dict()})

        t2 = time.perf_counter()
        out = await self._dispatch(ctx, session, utterance, clean, decision)
        timings["flow_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        text = " ".join(r for r in out.replies if r)
        session.last_bot = text
        if out.transfer:
            session.transfer = out.transfer
            session.ended = True
        if out.end_call:
            session.ended = True
        res = TurnResult(out.replies, text, decision.as_dict(), events, self.snapshot(session), timings,
                         ended=session.ended, transfer=session.transfer)
        self._log(session, utterance, text, decision.as_dict(), events)
        return res

    # ── routing context ──
    def _route_context(self, ctx: TurnCtx, session: Session, utterance: str, clean: str) -> RouteContext:
        run = session.active
        wf = self.workflows.get(run.name) if run else None
        faq = self.faq.match(clean)
        mentions = any(r.status in ("resolved", "ambiguous") or (r.status == "confirm" and r.best.score >= 0.8)
                       for r in self.kit.resolve_many("test", utterance))
        rc = RouteContext(
            utterance=utterance, text=clean, active=run.name if run else None,
            step=run.step if run else None,
            expects=wf.expects(run) if wf and run else "text",
            pending_kind=run.pending.kind if run and run.pending else None,
            session_pending=session.pending.kind if session.pending else None,
            suspended=[r.name for r in session.suspended], last_bot=session.last_bot,
            workflows=self.manifests,
            can_interrupt=wf.can_interrupt(run) if wf and run else True,
            has_progress=wf.has_progress(run) if wf and run else False,
            mentions_test=mentions,
            slot_answer=bool(wf and run and wf.answers_current(ctx, run, utterance)),
            faq_status=faq.status, faq_score=faq.score,
        )
        return rc

    # ── dispatch ──
    async def _dispatch(self, ctx: TurnCtx, session: Session, utt: str, clean: str, d: RouteDecision) -> TurnOut:
        dec = d.decision
        if dec == "human":
            return TurnOut(["Theek hai, main aap ko hamare saathi se mila deti hoon. Kripya line par rahein."],
                           transfer="caller_requested")
        if dec == "end":
            return await self._end(ctx, session)
        if session.pending and dec == "continue":
            return await self._handle_session_pending(ctx, session, utt, clean)
        if dec == "faq":
            return self._faq(ctx, session, clean)
        if dec == "lookup":
            return await self._lookup(ctx, session, utt)
        if dec == "switch":
            return await self._switch(ctx, session, utt, d.workflow)
        if dec == "resume":
            return await self._resume(ctx, session)
        if dec == "unclear":
            return self._unclear(ctx, session, clean)
        # continue
        if session.active:
            session.unclear_count = 0
            return await self._run_workflow(ctx, session, utt)
        return self._unclear(ctx, session, clean)

    async def _run_workflow(self, ctx: TurnCtx, session: Session, utt: str) -> TurnOut:
        run = session.active
        wf = self.workflows[run.name]
        out = await wf.handle(ctx, session, run, utt)
        return await self._after_workflow(ctx, session, run, out)

    async def _after_workflow(self, ctx: TurnCtx, session: Session, run: WorkflowRun, out: TurnOut) -> TurnOut:
        if out.transfer:
            return out
        if out.done:
            session.active = None
            title = self.workflows[run.name].title
            if session.suspended:
                nxt = session.suspended[-1]
                q = self._p("Ab kya {title} wahin se jari rakhein?", title=self.workflows[nxt.name].spoken)
                session.pending = Pending("resume", {"name": nxt.name}, q)
                out.say(q)
            else:
                out.say(self._p("Kya main aur kisi cheez mein madad kar {sakti} hoon?"))
        return out

    # ── session-level pending ──
    async def _handle_session_pending(self, ctx: TurnCtx, session: Session, utt: str, clean: str) -> TurnOut:
        p = session.pending
        yn = nz.yes_no(utt)
        out = TurnOut()
        if p.kind == "switch":
            session.pending = None
            if yn == "yes":
                return await self._do_switch(ctx, session, p.data["target"], p.data.get("utterance"))
            if yn == "no":
                out.say("Theek hai, pehle wali baat jari rakhte hain.")
                return self._reask(ctx, session, out)
            # not a yes/no: treat as an answer for the active task
            return await self._run_workflow(ctx, session, utt)
        if p.kind == "resume":
            session.pending = None
            if yn == "yes" and session.suspended:
                return await self._do_resume(ctx, session)
            if yn == "no":
                session.suspended.pop()
                out.say(self._p("Theek hai, us ko chhod dete hain. Kya main aur kisi cheez mein madad kar {sakti} hoon?"))
                return out
            return self._unclear(ctx, session, clean)
        if p.kind == "end_confirm":
            session.pending = None
            if yn == "yes":
                out.say("Theek hai. Allah hafiz.")
                out.end_call = True
                return out
            out.say("Theek hai, jari rakhte hain.")
            return self._reask(ctx, session, out)
        if p.kind == "faq_confirm":
            session.pending = None
            if yn == "yes":
                out.say(p.data["answer"])
                if p.data.get("handoff"):
                    out.transfer = f"faq_{p.data['id'].removeprefix('faq_')}"
                    return out
                return self._reask(ctx, session, out)
            out.say("Maaf kijiye. Aap apna sawal dobara poochh sakte hain.")
            return self._reask(ctx, session, out)
        if p.kind in ("lookup_choose", "lookup_confirm"):
            lrun = session.lookup
            res = await sl.resolve_pending(ctx, session, lrun, utt)
            return self._lookup_finish(ctx, session, lrun, res, out)
        session.pending = None
        return self._unclear(ctx, session, clean)

    def _reask(self, ctx: TurnCtx, session: Session, out: TurnOut) -> TurnOut:
        if session.active:
            wf = self.workflows[session.active.name]
            out.say(wf.reask(ctx, session, session.active))
        return out

    # ── FAQ / lookup ──
    def _faq(self, ctx: TurnCtx, session: Session, clean: str) -> TurnOut:
        m = self.faq.match(clean)
        out = TurnOut()
        if m.status == "answer":
            ctx.emit("tool", {"name": "faq", "id": m.entry["id"], "score": round(m.score, 3)})
            out.say(m.text)
            if m.entry.get("handoff"):
                out.transfer = f"faq_{m.entry['id'].removeprefix('faq_')}"
                return out
        elif m.status == "confirm":
            q = self._p("Kya aap yeh poochh {rahi} hain: {question}?", question=m.entry["questions"][0])
            session.pending = Pending("faq_confirm", {"answer": m.text, "handoff": bool(m.entry.get("handoff")),
                                                      "id": m.entry["id"]}, q)
            out.say(q)
            return out
        else:
            out.say("Maaf kijiye, mujhe is ke baare mein pakka maloom nahi. Kya aap hamare saathi se baat karna chahenge?")
        return self._reask(ctx, session, out)

    async def _lookup(self, ctx: TurnCtx, session: Session, utt: str) -> TurnOut:
        lrun = WorkflowRun("_lookup")
        session.lookup = lrun
        mentions = self._test_mentions_for_lookup(utt)
        res = sl._process_mentions(ctx, lrun, mentions, "test", "lookup_items")
        return self._lookup_finish(ctx, session, lrun, res, TurnOut())

    def _test_mentions_for_lookup(self, utt: str) -> list[str]:
        words = [w for w in nz.clean_text(utt).split() if w not in
                 {"price", "qeemat", "kimat", "rate", "kitne", "kitna", "kitni", "ka", "ki", "ke", "kya", "hai", "hain",
                  "charges", "fees", "cost", "bata", "batao", "dein", "do", "batayen", "paise", "rupay", "kharcha", "mujhe"}]
        return self.kit.split_mentions("test", " ".join(words))

    def _lookup_finish(self, ctx: TurnCtx, session: Session, lrun: WorkflowRun, res: sl.SlotOutcome | None,
                       out: TurnOut) -> TurnOut:
        if res is None:
            session.pending = None
            return self._reask(ctx, session, out)
        out.say(*res.say)
        if res.transfer:
            out.transfer = res.transfer
            session.pending = None
            return out
        if res.asked:
            kind = lrun.pending.kind if lrun.pending else "lookup_choose"
            session.pending = Pending("lookup_choose" if kind != "confirm_entity" else "lookup_confirm", {}, res.say[-1] if res.say else "")
            return out
        session.pending = None
        items = lrun.slots.get("lookup_items", [])
        if not items:
            out.say("Maaf kijiye, test ka naam samajh nahi aaya. Dobara bata dein.")
            session.pending = Pending("lookup_choose", {}, "")
            lrun.pending = Pending("spell", {"entity": "test", "slot": "lookup_items", "rest": [], "mention": "", "rounds": 0}, "")
            return out
        for it in items:
            f = ctx.kit.facts("test", it["id"], ["price", "fasting", "fasting_hours"])
            price = f.get("price", "")
            if price.isdigit():
                line = f"{it['name']} ki price {nz.speak_number(int(price))} rupay hai."
            else:
                line = f"{it['name']} ki price abhi maloom nahi."
            if f.get("fasting", "").lower() == "yes":
                line += f" Is ke liye {f.get('fasting_hours') or 8} ghante kuch khana peena mana hai."
            out.say(line)
        ctx.emit("tool", {"name": "lookup_test_price", "items": [i["name"] for i in items]})
        return self._reask(ctx, session, out)

    # ── switching ──
    async def _switch(self, ctx: TurnCtx, session: Session, utt: str, target: str | None) -> TurnOut:
        out = TurnOut()
        if target is None or target not in self.workflows:
            return self._unclear(ctx, session, nz.clean_text(utt))
        run = session.active
        if run and run.name == target:
            return await self._run_workflow(ctx, session, utt)
        if run:
            wf = self.workflows[run.name]
            if not wf.can_interrupt(run):
                out.say("Pehle yeh mukammal kar lete hain, phir aap ki doosri baat dekhte hain.")
                return self._reask(ctx, session, out)
            if len(session.suspended) >= self.s.max_suspended:
                out.say("Maaf kijiye, itne kaam ek saath nahi ho sakte. Main aap ko hamare saathi se mila deti hoon.")
                out.transfer = "too_many_switches"
                return out
            if wf.has_progress(run):
                q = self._p("Aap {a} chhod kar {b} shuru karna chahte hain?",
                            a=wf.spoken, b=self.workflows[target].spoken)
                session.pending = Pending("switch", {"target": target, "utterance": utt}, q)
                out.say(q)
                return out
            session.active = None     # nothing collected yet: just replace
        return await self._do_switch(ctx, session, target, utt)

    async def _do_switch(self, ctx: TurnCtx, session: Session, target: str, utt: str | None) -> TurnOut:
        run = session.active
        if run:
            run.status = "suspended"
            session.suspended.append(run)
        new = self.workflows[target].new_run()
        session.active = new
        out = await self.workflows[target].start(ctx, session, new, utt)
        return await self._after_workflow(ctx, session, new, out)

    async def _resume(self, ctx: TurnCtx, session: Session) -> TurnOut:
        if not session.suspended:
            return self._unclear(ctx, session, "")
        run = session.active
        if run:
            run.status = "suspended"
            session.suspended.append(run)
            session.active = None
        return await self._do_resume(ctx, session)

    async def _do_resume(self, ctx: TurnCtx, session: Session) -> TurnOut:
        run = session.suspended.pop()
        run.status = "active"
        session.active = run
        wf = self.workflows[run.name]
        out = TurnOut()
        out.say(f"Theek hai, {wf.spoken} wahin se jari rakhte hain.", wf.reask(ctx, session, run))
        return out

    # ── unclear / end ──
    def _unclear(self, ctx: TurnCtx, session: Session, clean: str) -> TurnOut:
        out = TurnOut()
        session.unclear_count += 1
        if session.unclear_count >= 3:
            out.say("Maaf kijiye, main samajh nahi pa rahi. Main aap ko hamare saathi se mila deti hoon.")
            out.transfer = "unclear"
            return out
        if not session.active and any(g in f" {clean} " for g in (" assalam", " salam", " hello", " hi ", " salaam")):
            out.say("Walaikum assalam! Aap home sampling booking karna chahte hain, ya kisi booking ka status dekhna chahte hain?")
            return out
        out.say("Maaf kijiye, samajh nahi aaya.")
        if session.active:
            return self._reask(ctx, session, out)
        out.say("Aap home sampling booking karna chahte hain, ya kisi booking ka status dekhna chahte hain?")
        return out

    async def _end(self, ctx: TurnCtx, session: Session) -> TurnOut:
        out = TurnOut()
        run = session.active
        if run and self.workflows[run.name].has_progress(run) and run.status == "active":
            q = "Aap ki baat abhi mukammal nahi hui. Kya phir bhi call khatam karni hai?"
            session.pending = Pending("end_confirm", {}, q)
            out.say(q)
            return out
        out.say("Shukriya, Chughtai Lab se rabta karne ka. Allah hafiz.")
        out.end_call = True
        return out

    # ── logging (PII-redacted by default) ──
    def _log(self, session: Session, utt: str, bot: str, route: dict, events: list[dict]) -> None:
        try:
            self.s.log_dir.mkdir(parents=True, exist_ok=True)
            red = self.s.redact_pii
            rec = {
                "ts": datetime.now().isoformat(timespec="seconds"), "session": session.id, "turn": session.turn,
                "user": _redact(utt) if red else utt, "bot": _redact(bot) if red else bot,
                "route": route if self.s.router_log else {},
                "state": {"active": session.active.name if session.active else None,
                          "step": session.active.step if session.active else None,
                          "slots": sorted(session.active.slots) if (session.active and red) else
                          (session.active.slots if session.active else {})},
            }
            with (self.s.log_dir / f"{datetime.now():%Y%m%d}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


def _redact(text: str) -> str:
    import re
    return re.sub(r"\d", "#", text or "")
