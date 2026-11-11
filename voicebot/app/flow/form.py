"""Workflow engine: a workflow is a folder (workflow.yaml, manifest.yaml, optional hooks.py, prompts/, scenarios/).

workflow.yaml declares slots and ordered steps; hooks.py holds the few pieces that are real code
(verification, submit, fasting rules). The engine fills slots opportunistically, handles the shared
spoken sub-dialogues (choose / confirm / spell) and read-back, and tells the bot what to say next.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.flow import slots as sl
from app.flow.types import Pending, Session, TurnCtx, TurnOut, WorkflowRun, fmt
from app.nlp import normalizers as nz

_SLOT_KEYWORDS = {
    "tests": {"test", "tests", "sugar", "cbc"},
    "patient_name": {"naam", "name", "mareez"},
    "age": {"umar", "age", "saal"},
    "gender": {"jins", "gender"},
    "address": {"address", "pata", "jagah", "area", "ghar", "location"},
    "date": {"waqt", "din", "date", "time", "tareekh", "subah", "shaam", "dopahar", "raat", "kal", "parson"},
    "phone": {"number", "phone", "rabta", "mobile", "contact"},
}


@dataclass
class HookResult:
    say: list[str] = field(default_factory=list)
    ok: bool = True
    clear: list[str] = field(default_factory=list)   # slots to clear (caller must re-enter)
    transfer: str | None = None
    done: bool = False                                # workflow finished
    end_call: bool = False


class Workflow:
    def __init__(self, folder: Path):
        self.folder = folder
        self.spec: dict = yaml.safe_load((folder / "workflow.yaml").read_text(encoding="utf-8"))
        self.manifest: dict = yaml.safe_load((folder / "manifest.yaml").read_text(encoding="utf-8"))
        self.name: str = self.spec["name"]
        self.title: str = self.spec.get("title", self.name)
        self.spoken: str = self.spec.get("spoken", self.title.lower())
        self.steps: list[dict] = self.spec["steps"]
        self.slot_types: dict[str, str] = {k: v["type"] for k, v in self.spec["slots"].items()}
        self.hooks = self._load_hooks()

    def _load_hooks(self):
        p = self.folder / "hooks.py"
        if not p.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"wf_hooks_{self.name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _hook(self, name: str):
        return getattr(self.hooks, name, None) if self.hooks else None

    # ── helpers ──
    def new_run(self) -> WorkflowRun:
        return WorkflowRun(name=self.name)

    def _step(self, sid: str) -> dict | None:
        return next((s for s in self.steps if s["id"] == sid), None)

    def _when(self, step: dict, run: WorkflowRun) -> bool:
        w = step.get("when")
        if not w:
            return True
        v = run.slots.get(w.get("slot"))
        if "eq" in w:
            return v == w["eq"]
        if "in" in w:
            return v in w["in"]
        if "filled" in w:
            return w["filled"] in run.slots
        return True

    def vars(self, ctx: TurnCtx, session: Session, run: WorkflowRun) -> dict[str, Any]:
        s = run.slots
        v: dict[str, Any] = dict(ctx.persona)
        v.update({k: val for k, val in s.items() if isinstance(val, (str, int))})
        v["tests_spoken"] = sl.spoken_tests(run)
        v["patient_name"] = s.get("patient_name", "")
        v["age_spoken"] = f"{s['age']} saal" if s.get("age") else ""
        if s.get("date"):
            v["when_spoken"] = sl.spoken_when(s["date"], ctx.today)
        if isinstance(s.get("address"), dict):
            a = s["address"]
            v["address_spoken"] = a["text"] if len(a["text"]) < 90 else f"{a['area']}"
            v["area"] = a["area"]
            v["home_fee"] = a.get("home_fee", "")
        if s.get("phone"):
            v["phone_spoken"] = nz.speak_digits(s["phone"])
        h = self._hook("render_vars")
        if h:
            v.update(h(ctx, session, run) or {})
        return v

    def _render(self, template: str | list, ctx: TurnCtx, session: Session, run: WorkflowRun, idx: int = 0) -> str:
        if isinstance(template, list):
            template = template[min(idx, len(template) - 1)]
        return fmt(template, **self.vars(ctx, session, run))

    def can_interrupt(self, run: WorkflowRun) -> bool:
        if run.pending and run.pending.kind == "confirm_step":
            sid = run.pending.data.get("step")
            st = self._step(sid) if sid else None
            if st and st.get("non_interruptible"):
                return False
        return True

    def has_progress(self, run: WorkflowRun) -> bool:
        return bool(run.slots)

    def summary(self, ctx: TurnCtx, session: Session, run: WorkflowRun) -> dict:
        return {"workflow": self.name, "step": run.step, "slots": run.slots, "status": run.status}

    # ── start ──
    def _wire_veto(self, ctx: TurnCtx, run: WorkflowRun) -> None:
        """Let slot parsers ask the workflow whether a resolved entity is acceptable (e.g. imaging cannot be done at home)."""
        h = self._hook("veto_entity")
        if h:
            run.scratch["veto"] = lambda entity, rid: h(ctx, entity, rid)
            run.scratch["veto_note"] = self.spec.get("veto_note", "{names} is workflow mein shamil nahi ho sakta.")

    async def start(self, ctx: TurnCtx, session: Session, run: WorkflowRun, text: str | None = None) -> TurnOut:
        self._wire_veto(ctx, run)
        out = TurnOut()
        if self.spec.get("intro"):
            out.say(self._render(self.spec["intro"], ctx, session, run))
        h = self._hook("on_start")
        if h:
            r = h(ctx, session, run)
            if r:
                out.say(*r)
        # a first utterance (e.g. "mujhe cbc karwana hai") may already contain slot values
        if text:
            await self._opportunistic(text, ctx, session, run, exclude=None, first=True)
            first = self._first_step_with_input(run)
            if first and first.get("slot") and not self._filled(run, first["slot"]):
                run.step = first["id"]
                parser = sl.PARSERS[self.slot_types[first["slot"]]]
                if self.slot_types[first["slot"]] == "tests":
                    outcome = await sl.parse_tests(text, ctx, session, run, skip_none=True)
                else:
                    outcome = await parser(text, ctx, session, run)
                self._clean_empty(run)
                if outcome.asked or outcome.transfer or outcome.ok or outcome.note == "vetoed":
                    r = self._after_outcome(first, outcome, ctx, session, run, out)
                    if r is not None:
                        return r
                # nothing usable in the first sentence: just ask normally (not a failed attempt)
        return await self._advance(ctx, session, run, out)

    @staticmethod
    def _filled(run: WorkflowRun, slot: str) -> bool:
        return run.slots.get(slot) not in (None, "", [], {})

    @staticmethod
    def _clean_empty(run: WorkflowRun) -> None:
        for k in [k for k, v in run.slots.items() if v in ([], "", None, {})]:
            run.slots.pop(k, None)

    def _first_step_with_input(self, run: WorkflowRun) -> dict | None:
        for st in self.steps:
            if st.get("type", "slot") == "slot" and self._when(st, run):
                return st
        return None

    # ── turn ──
    async def handle(self, ctx: TurnCtx, session: Session, run: WorkflowRun, text: str) -> TurnOut:
        self._wire_veto(ctx, run)
        out = TurnOut()
        p = run.pending
        if p is not None:
            r = await self._handle_pending(p, text, ctx, session, run, out)
            if r is not None:
                return r
        step = self._step(run.step) if run.step else None
        if step is None or step.get("type", "slot") != "slot":
            return await self._advance(ctx, session, run, out)
        r = await self._try_fill(step, text, ctx, session, run, out)
        if r is not None:
            return r
        return await self._advance(ctx, session, run, out)

    async def _try_fill(self, step: dict, text: str, ctx, session, run, out: TurnOut) -> TurnOut | None:
        slot = step["slot"]
        parser = sl.PARSERS[self.slot_types[slot]]
        outcome = await parser(text, ctx, session, run)
        if not outcome.ok and not outcome.asked:
            self._clean_empty(run)
        await self._opportunistic(text, ctx, session, run, exclude=slot)
        return self._after_outcome(step, outcome, ctx, session, run, out)

    def _after_outcome(self, step, outcome: sl.SlotOutcome, ctx, session, run, out: TurnOut) -> TurnOut | None:
        out.say(*outcome.say)
        if outcome.transfer:
            out.transfer = outcome.transfer
            return out
        if outcome.asked:
            run.last_ask = run.pending.prompt if run.pending else ""
            return out
        if outcome.ok:
            slot = step["slot"]
            self._share_facts(session, run, slot)
            run.retries.pop(step["id"], None)
            if step.get("ack"):
                out.say(self._render(step["ack"], ctx, session, run))
            return None   # continue to advance
        if outcome.note == "vetoed":      # everything the caller named was rejected by a workflow rule (already explained)
            out.say(self._ask_text(step, ctx, session, run))
            run.last_ask = out.replies[-1] if out.replies else ""
            return out
        # not understood
        n = run.retries.get(step["id"], 0) + 1
        run.retries[step["id"]] = n
        if n > ctx.settings.max_slot_retries:
            out.say("Maaf kijiye, main yeh sahi se samajh nahi pa rahi. Main aap ko hamare saathi se mila deti hoon.")
            out.transfer = f"slot_failed_{step['id']}"
            return out
        if outcome.note in ("need_part", "need_day", "not_covered", "past_date", "slot_unavailable") or outcome.say:
            pass
        out.say(self._ask_text(step, ctx, session, run, retry=n, note=outcome.note))
        run.last_ask = out.replies[-1] if out.replies else ""
        return out

    def _share_facts(self, session: Session, run: WorkflowRun, slot: str) -> None:
        if slot == "patient_name" and run.slots.get("patient_name"):
            session.facts.setdefault("name", run.slots["patient_name"])

    def _ask_text(self, step, ctx, session, run, retry: int = 0, note: str = "") -> str:
        if note == "need_part" and step.get("ask_part"):
            day = run.scratch.get("date_day")
            from datetime import date as _d
            delta = (_d.fromisoformat(day) - ctx.today).days if day else 1
            lab = {0: "aaj", 1: "kal", 2: "parson"}.get(delta, day)
            return fmt(step["ask_part"], day_label=lab, **ctx.persona)
        if note == "need_day" and step.get("ask_day"):
            return fmt(step["ask_day"], **ctx.persona)
        key = "retry" if retry and step.get("retry") else "ask"
        return self._render(step.get(key, step.get("ask", "")), ctx, session, run, max(retry - 1, 0))

    # ── opportunistic filling ──
    async def _opportunistic(self, text: str, ctx, session, run, exclude: str | None, first: bool = False) -> None:
        toks = set(nz.tokens(text))
        for slot, typ in self.slot_types.items():
            if slot == exclude or self._filled(run, slot):
                continue
            try:
                if typ == "age" and toks & {"umar", "saal", "age", "years", "year"}:
                    r = await sl.parse_age(text, ctx, session, run)
                elif typ == "gender":
                    r = await sl.parse_gender(text, ctx, session, run)
                elif typ == "datetime":
                    s = nz.resolve_date(text, ctx.today)
                    if not (s.day and s.part):
                        continue
                    r = await sl.parse_datetime(text, ctx, session, run)
                elif typ == "phone":
                    res = nz.normalize_phone(text)
                    if res.valid and len(nz.extract_digits(text)) >= 10:
                        run.scratch["phone_hint"] = res.digits
                    continue
                elif typ == "tests" and first:
                    res_list = ctx.kit.resolve_many("test", text)
                    good = [r_ for r_ in res_list if r_.status == "resolved"]
                    veto = run.scratch.get("veto")
                    if veto and any(r_.candidates and veto("test", r_.candidates[0].id) for r_ in res_list):
                        continue            # let the main parser explain what cannot be done in this workflow
                    if good and len(good) == len(res_list):
                        for g in good:
                            run.slots.setdefault("tests", []).append({"id": g.best.id, "name": g.best.name})
                    continue
                else:
                    continue
            except Exception:
                continue
            _ = r

    # ── pending handling ──
    async def _handle_pending(self, p: Pending, text: str, ctx, session, run, out: TurnOut) -> TurnOut | None:
        kind = p.kind
        outcome: sl.SlotOutcome | None = None
        if kind in ("choose", "confirm_entity", "spell"):
            outcome = await sl.resolve_pending(ctx, session, run, text)
        elif kind in ("choose_area", "house") or (kind == "confirm_slot" and p.data.get("slot") == "address"):
            outcome = await sl.resolve_area_pending(ctx, session, run, text)
        elif kind == "confirm_slot" and p.data.get("slot") == "phone":
            outcome = await sl.resolve_phone_pending(ctx, session, run, text)
        elif kind == "confirm_step":
            return await self._handle_confirm_step(p, text, ctx, session, run, out)
        elif kind == "which_slot":
            return await self._handle_which_slot(p, text, ctx, session, run, out)
        if outcome is None:
            run.pending = None
            return None
        step = self._step(run.step) if run.step else None
        # a pending always belongs to the step that created it, even if that step is not the current one
        owner = self._owner_step(kind, p) or step
        if owner is None:
            return await self._advance(ctx, session, run, out) if outcome.ok else out.say(*outcome.say)
        r = self._after_outcome(owner, outcome, ctx, session, run, out)
        if r is not None:
            return r
        return await self._advance(ctx, session, run, out)

    def _owner_step(self, kind: str, p: Pending) -> dict | None:
        slot = p.data.get("slot")
        if kind in ("choose_area", "house"):
            slot = "address"
        if slot:
            return next((s for s in self.steps if s.get("slot") == slot), None)
        return None

    async def _handle_confirm_step(self, p: Pending, text: str, ctx, session, run, out: TurnOut) -> TurnOut:
        step = self._step(p.data["step"])
        yn = nz.yes_no(text)
        if yn == "yes":
            run.scratch[f"{step['id']}_done"] = True
            run.pending = None
            if step["id"] == "confirm":
                run.confirmed = True
            return await self._advance(ctx, session, run, out)
        if yn == "no":
            # did the caller already say what is wrong? ("nahi, kal nahi parson")
            changed = await self._apply_corrections(text, ctx, session, run)
            run.pending = None
            if changed:
                out.say("Theek hai, tabdeel kar diya.")
                run.scratch.pop(f"{step['id']}_done", None)
                run.confirmed = False
                return await self._advance(ctx, session, run, out)
            if step.get("on_no") == "reask_date":
                run.slots.pop("date", None)
                run.scratch.pop(f"{step['id']}_done", None)
                out.say("Koi baat nahi, doosra waqt chun lete hain.")
                return await self._advance(ctx, session, run, out)
            q = "Kya theek nahi hai? Naam, tests, address, waqt ya number?"
            run.pending = Pending("which_slot", {"step": step["id"]}, q)
            out.say(q)
            return out
        changed = await self._apply_corrections(text, ctx, session, run)
        if changed:
            run.pending = None
            run.confirmed = False
            run.scratch.pop(f"{step['id']}_done", None)
            out.say("Theek hai, tabdeel kar diya.")
            return await self._advance(ctx, session, run, out)
        out.say("Maaf kijiye, sirf haan ya nahi bata dein.", p.prompt)
        return out

    async def _handle_which_slot(self, p: Pending, text: str, ctx, session, run, out: TurnOut) -> TurnOut:
        toks = set(nz.tokens(text))
        for slot, kws in _SLOT_KEYWORDS.items():
            if slot in run.slots and toks & kws and slot in self.slot_types:
                run.slots.pop(slot, None)
                run.scratch.pop("phone_buf", None)
                run.confirmed = False
                run.scratch.pop(f"{p.data['step']}_done", None)
                run.pending = None
                if slot == "date":
                    run.scratch.pop("date_day", None)
                    run.scratch.pop("date_part", None)
                out.say("Theek hai.")
                return await self._advance(ctx, session, run, out)
        changed = await self._apply_corrections(text, ctx, session, run)
        if changed:
            run.pending = None
            run.confirmed = False
            run.scratch.pop(f"{p.data['step']}_done", None)
            out.say("Theek hai, tabdeel kar diya.")
            return await self._advance(ctx, session, run, out)
        out.say("Maaf kijiye, samajh nahi aaya. Naam, tests, address, waqt ya number mein se kya galat hai?")
        return out

    async def _apply_corrections(self, text: str, ctx, session, run) -> bool:
        changed = False
        s = nz.resolve_date(text, ctx.today)
        if "date" in self.slot_types and (s.day or s.part):
            old = run.slots.get("date")
            if old:
                run.scratch["date_day"] = old["day"]
                run.scratch["date_part"] = old["part"]
            r = await sl.parse_datetime(text, ctx, session, run)
            new = run.slots.get("date")
            if r.ok and new != old and (not old or (new["day"], new["part"]) != (old["day"], old["part"])):
                changed = True
            elif old and not r.ok:
                run.slots["date"] = old      # keep the old date if the correction could not be applied
        res = nz.normalize_phone(text)
        if "phone" in self.slot_types and res.valid and len(nz.extract_digits(text)) >= 10:
            if run.slots.get("phone") != res.digits:
                run.slots["phone"] = res.digits
                changed = True
        if "age" in self.slot_types and {"umar", "saal"} & set(nz.tokens(text)):
            a = nz.extract_age(text)
            if a and run.slots.get("age") != a:
                run.slots["age"] = a
                changed = True
        return changed

    # ── advance ──
    def _next_step(self, run: WorkflowRun) -> dict | None:
        for st in self.steps:
            if not self._when(st, run):
                continue
            typ = st.get("type", "slot")
            if typ == "slot":
                if not self._filled(run, st["slot"]):
                    return st
            elif typ in ("confirm", "hook_confirm"):
                if not run.scratch.get(f"{st['id']}_done"):
                    return st
            elif typ == "action":
                if not run.scratch.get(f"{st['id']}_done"):
                    return st
        return None

    async def _advance(self, ctx: TurnCtx, session: Session, run: WorkflowRun, out: TurnOut) -> TurnOut:
        for _ in range(12):   # actions may chain
            st = self._next_step(run)
            if st is None:
                run.status = "done"
                out.done = True
                return out
            run.step = st["id"]
            typ = st.get("type", "slot")
            if typ == "slot":
                return self._ask_slot(st, ctx, session, run, out)
            if typ == "confirm":
                text = self._render(st["summary"], ctx, session, run)
                run.pending = Pending("confirm_step", {"step": st["id"]}, text)
                out.say(text)
                return out
            if typ == "hook_confirm":
                h = self._hook(st["hook"])
                text = h(ctx, session, run) if h else None
                if not text:
                    run.scratch[f"{st['id']}_done"] = True
                    continue
                run.pending = Pending("confirm_step", {"step": st["id"]}, text)
                out.say(text)
                return out
            if typ == "action":
                res: HookResult = await self._hook(st["hook"])(ctx, session, run)
                run.scratch[f"{st['id']}_done"] = True
                ctx.emit("tool", {"name": st["hook"], "ok": res.ok})
                out.say(*res.say)
                for sname in res.clear:
                    run.slots.pop(sname, None)
                if res.transfer:
                    out.transfer = res.transfer
                    return out
                if res.end_call:
                    out.end_call = True
                if res.done:
                    run.status = "done"
                    out.done = True
                    return out
                if not res.ok:
                    run.scratch.pop(f"{st['id']}_done", None)
                    if res.clear:
                        continue
                    return out
                continue
        return out

    def _ask_slot(self, st: dict, ctx, session, run, out: TurnOut) -> TurnOut:
        slot = st["slot"]
        typ = self.slot_types[slot]
        # phone: confirm the caller's own number / a number already heard instead of asking from scratch
        if typ == "phone":
            hint = run.scratch.get("phone_hint") or (session.facts.get("caller_phone") if st.get("use_caller_id") else None) \
                or (session.facts.get("phone") if st.get("prefill") else None)
            if hint and not run.scratch.get("phone_hint_asked"):
                run.scratch["phone_hint_asked"] = True
                q = f"Kya aap isi number {nz.speak_digits(hint)} par rabta chahte hain?"
                run.pending = Pending("confirm_slot", {"slot": "phone", "value": hint}, q)
                out.say(q)
                return out
        q = self._ask_text(st, ctx, session, run, retry=run.retries.get(st["id"], 0),
                           note=("need_part" if run.scratch.get("date_day") and typ == "datetime" else
                                 "need_day" if run.scratch.get("date_part") and typ == "datetime" else ""))
        run.last_ask = q
        out.say(q)
        return out

    def reask(self, ctx: TurnCtx, session: Session, run: WorkflowRun) -> str:
        """Text to repeat after an interruption (FAQ, lookup)."""
        if run.pending and run.pending.prompt:
            return run.pending.prompt
        st = self._step(run.step) if run.step else None
        if st and st.get("type", "slot") == "slot":
            return self._ask_text(st, ctx, session, run, retry=0)
        return run.last_ask

    def answers_current(self, ctx: TurnCtx, run: WorkflowRun, text: str) -> bool:
        """Pure check: does `text` look like an answer to the question currently asked? (no state change)"""
        toks = set(nz.tokens(text))
        p = run.pending
        if p is not None:
            k = p.kind
            if k in ("choose", "choose_area"):
                cands = p.data.get("cands", [])
                ent = p.data.get("entity", "area")
                return bool(toks & sl._BOTH) or sl._ordinal_index(text, len(cands)) is not None or \
                    sl._match_label(ctx, ent, text, cands) is not None
            if k in ("confirm_entity", "confirm_slot", "confirm_step"):
                return nz.yes_no(text) is not None
            return k in ("spell", "house", "which_slot")
        st = self._step(run.step) if run.step else None
        if not st or st.get("type", "slot") != "slot":
            return False
        typ = self.slot_types[st["slot"]]
        if typ == "age":
            return nz.extract_age(text) is not None
        if typ == "gender":
            return nz.extract_gender(text) is not None
        if typ == "phone":
            return len(nz.extract_digits(text)) >= 3
        if typ == "datetime":
            s_ = nz.resolve_date(text, ctx.today)
            return bool(s_.day or s_.part or s_.hour)
        if typ == "booking_ref":
            return len(nz.extract_digits(text)) >= 4
        if typ == "tests":
            return any(r.status in ("resolved", "ambiguous") or (r.status == "confirm" and r.best.score >= 0.8)
                       for r in ctx.kit.resolve_many("test", text))
        if typ == "address":
            r_ = ctx.kit.resolve("area", text)
            return r_.status in ("resolved", "ambiguous") or (r_.status == "confirm" and r_.best.score >= 0.8)
        if typ == "action":
            toks_ = toks
            return bool(toks_ & ({"status", "cancel", "reschedule", "badalna", "badalni", "badlo"}))
        return False

    def expects(self, run: WorkflowRun) -> str:
        """What kind of answer the bot is waiting for (used by the router's cheap pass)."""
        if run.pending:
            k = run.pending.kind
            if k in ("confirm_entity", "confirm_step") or (k == "confirm_slot"):
                return "yes_no"
            if k in ("choose", "choose_area"):
                return "choice"
            if k == "spell":
                return "text"
            if k == "house":
                return "text"
            if k == "which_slot":
                return "text"
        st = self._step(run.step) if run.step else None
        if st and st.get("type", "slot") == "slot":
            t = self.slot_types[st["slot"]]
            return {"phone": "number", "age": "number", "datetime": "date", "gender": "text", "action": "text"}.get(t, "text")
        return "text"
