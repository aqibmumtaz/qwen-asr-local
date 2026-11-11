"""Slot parsers and the shared spoken sub-dialogues (choose / confirm / spell) used by every workflow.

Each parser returns a SlotOutcome. When a parser needs the caller to pick or confirm something it sets
`run.pending` and returns asked=True with the question in `say`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from app.flow.types import Pending, TurnCtx, WorkflowRun, Session, fmt
from app.nlp import normalizers as nz

_FIRST = {"pehla", "pehli", "pehle", "first", "1"}
_SECOND = {"doosra", "dusra", "doosri", "dusri", "second", "2"}
_THIRD = {"teesra", "teesri", "third", "3"}
_LAST = {"akhri", "aakhri", "last"}
_BOTH = {"dono", "both", "sab", "teeno", "sare", "sari", "saray"}
_CANT_SAY = {"pata", "maloom", "yaad"}
_PART_WORDS = {"subah", "dopahar", "shaam", "raat"}


@dataclass
class SlotOutcome:
    ok: bool = False
    say: list[str] = field(default_factory=list)
    asked: bool = False            # a follow-up question was asked (pending set)
    transfer: str | None = None
    note: str = ""


def _names(items: list[dict]) -> str:
    n = [i["name"] for i in items]
    if len(n) <= 1:
        return n[0] if n else ""
    return ", ".join(n[:-1]) + " aur " + n[-1]


def spoken_tests(run: WorkflowRun) -> str:
    return _names(run.slots.get("tests", []))


# ── tests ───────────────────────────────────────────────────────────────────
def _process_mentions(ctx: TurnCtx, run: WorkflowRun, mentions: list[str], entity: str, slot: str,
                      skip_none: bool = False) -> SlotOutcome:
    out = SlotOutcome()
    items: list[dict] = run.slots.setdefault(slot, [])
    mentions = list(mentions)
    veto = run.scratch.get("veto") if slot == "tests" else None
    vetoed: list[str] = []
    while mentions:
        m = mentions.pop(0)
        if len(ctx.kit.clean_mention(entity, m).replace(" ", "")) < 2:
            continue
        res = ctx.kit.resolve(entity, m)
        ctx.emit("tool", {"name": "resolve_entity", "args": {"entity": entity, "text": m}, "status": res.status,
                          "result": [(c.name, c.score) for c in res.candidates[:4]]})
        if veto and res.candidates:
            pool = res.candidates if res.status == "ambiguous" else res.candidates[:1]
            if all(veto(entity, c.id) for c in pool):      # the workflow cannot use any of them: explain, do not ask "which one?"
                vetoed.append(res.best.name if res.status != "ambiguous" else m.strip().upper() if len(m.strip()) <= 4 else m.strip())
                out.say.append(fmt(run.scratch.get("veto_note", "{names} nahi ho sakta."), names=vetoed[-1]))
                continue
        if res.status == "resolved":
            c = res.best
            if all(i["id"] != c.id for i in items):
                items.append({"id": c.id, "name": c.name})
        elif res.status == "confirm":
            c = res.best
            q = f"Kya aap {c.name} ke baare mein keh rahe hain?"
            run.pending = Pending("confirm_entity", {"entity": entity, "slot": slot, "cand": {"id": c.id, "name": c.name},
                                                     "rest": mentions, "mention": m}, q)
            out.say.append(q)
            out.asked = True
            return out
        elif res.status == "ambiguous":
            cands = [{"id": c.id, "name": c.name, "label": ctx.kit.describe(entity, c, [o for o in res.candidates if o is not c])}
                     for c in res.candidates]
            labels = _names([{"name": c["label"]} for c in cands])
            q = f"{m} ke {len(cands)} options hain: {labels}. Kaunsa chahiye?"
            if res.more:
                q = f"{m} ke kaafi options hain, jaise {labels}. Kaunsa chahiye?"
            run.pending = Pending("choose", {"entity": entity, "slot": slot, "cands": cands, "rest": mentions,
                                             "mention": m, "rounds": 0}, q)
            out.say.append(q)
            out.asked = True
            return out
        elif skip_none:
            continue
        else:
            q = f"'{m.strip()}' ka naam mujhe samajh nahi aaya. Dobara bata dein, ya spelling bata dein."
            run.pending = Pending("spell", {"entity": entity, "slot": slot, "rest": mentions, "mention": m, "rounds": 0}, q)
            out.say.append(q)
            out.asked = True
            return out
    out.ok = bool(items)
    if vetoed and not items:
        out.note = "vetoed"
    return out


async def parse_tests(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun, skip_none: bool = False) -> SlotOutcome:
    mentions = ctx.kit.split_mentions("test", text)
    out = _process_mentions(ctx, run, mentions, "test", "tests", skip_none=skip_none)
    if not out.ok and not out.asked and not out.note:
        out.note = "no_test_found"
    return out


# ── pending sub-dialogues ───────────────────────────────────────────────────
def _ordinal_index(text: str, n: int) -> int | None:
    toks = set(nz.tokens(text))
    if toks & _FIRST:
        return 0
    if toks & _SECOND and n > 1:
        return 1
    if toks & _THIRD and n > 2:
        return 2
    if toks & _LAST:
        return n - 1
    return None


def _match_label(ctx: TurnCtx, entity: str, text: str, cands: list[dict]) -> int | None:
    q = ctx.kit.clean_mention(entity, text) or text
    scores = []
    for i, c in enumerate(cands):
        names = [c["name"], c.get("label", "")]
        s = max(max(fuzz.token_set_ratio(q, n.lower()), fuzz.partial_ratio(q, n.lower())) for n in names if n)
        scores.append((s, i))
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 80 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 8):
        return scores[0][1]
    return None


async def resolve_pending(ctx: TurnCtx, session: Session, run: WorkflowRun, text: str) -> SlotOutcome | None:
    """Handle the answer to run.pending for entity sub-dialogues. Returns None if not an entity pending."""
    p = run.pending
    if p is None or p.kind not in ("choose", "confirm_entity", "spell"):
        return None
    d = p.data
    entity, slot = d["entity"], d["slot"]
    items: list[dict] = run.slots.setdefault(slot, [])
    rest = list(d.get("rest", []))
    yn = nz.yes_no(text)
    max_rounds = ctx.kit.max_rounds(entity)

    def done_with(new_items: list[dict]) -> SlotOutcome:
        for c in new_items:
            if all(i["id"] != c["id"] for i in items):
                items.append({"id": c["id"], "name": c["name"]})
        run.pending = None
        return _process_mentions(ctx, run, rest, entity, slot) if rest else SlotOutcome(ok=bool(items))

    def fail(retry_say: str) -> SlotOutcome:
        d["rounds"] = d.get("rounds", 0) + 1
        if d["rounds"] >= max_rounds:
            run.pending = None
            return SlotOutcome(say=["Maaf kijiye, main yeh sahi se samajh nahi pa rahi. Main aap ko hamare saathi se mila deti hoon."],
                               transfer=f"could_not_resolve_{entity}")
        run.pending = Pending(p.kind, d, retry_say)
        return SlotOutcome(say=[retry_say], asked=True)

    if p.kind == "confirm_entity":
        if yn == "yes":
            return done_with([d["cand"]])
        if yn == "no":
            d["rounds"] = d.get("rounds", 0) + 1
            run.pending = Pending("spell", {"entity": entity, "slot": slot, "rest": rest, "mention": d.get("mention", ""),
                                            "rounds": d["rounds"]}, "")
            q = "Theek hai, naam dobara bata dein, ya spelling bata dein."
            run.pending.prompt = q
            return SlotOutcome(say=[q], asked=True)
        # not yes/no: treat as a new mention
        run.pending = None
        return _process_mentions(ctx, run, ctx.kit.split_mentions(entity, text) + rest, entity, slot)

    if p.kind == "choose":
        cands = d["cands"]
        toks = set(nz.tokens(text))
        if toks & _BOTH and len(cands) > 1 and slot == "tests":
            return done_with(cands)
        idx = _ordinal_index(text, len(cands))
        if idx is None:
            idx = _match_label(ctx, entity, text, cands)
        if idx is not None:
            return done_with([cands[idx]])
        if yn == "no" and len(toks) <= 3:
            return fail("Theek hai, in mein se kaunsa chahiye? Pehla, doosra, ya naam bata dein.")
        # maybe the caller gave a fresh name instead
        res = ctx.kit.resolve(entity, text)
        if res.status in ("resolved", "confirm") and res.best and res.best.id not in {c["id"] for c in cands}:
            run.pending = None
            return _process_mentions(ctx, run, ctx.kit.split_mentions(entity, text) + rest, entity, slot)
        return fail("Maaf kijiye, samajh nahi aaya. Pehla ya doosra bata dein, ya poora naam bolein.")

    # spell
    res = ctx.kit.resolve_spelled(entity, text)
    ctx.emit("tool", {"name": "resolve_entity", "args": {"entity": entity, "text": text, "spelled": True},
                      "status": res.status, "result": [(c.name, c.score) for c in res.candidates[:4]]})
    if res.status == "resolved":
        c = res.best
        return done_with([{"id": c.id, "name": c.name}])
    if res.status in ("confirm", "ambiguous"):
        run.pending = None
        return _process_mentions(ctx, run, [text] + rest, entity, slot)
    return fail("Naam abhi bhi samajh nahi aaya. Ek ek harf bol kar spelling bata dein.")


# ── name / age / gender ─────────────────────────────────────────────────────
async def parse_name(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    name = None
    if ctx.llm and ctx.llm.enabled:
        j = await ctx.llm.chat_json([
            {"role": "system", "content": "Extract the patient's personal name from a noisy Roman Urdu phone transcript. "
             'Reply with JSON only: {"name": "<name or null>"}. Do not include titles, ages or filler words.'},
            {"role": "user", "content": text}], max_tokens=60)
        if j and isinstance(j.get("name"), str) and j["name"].strip().lower() not in ("", "null", "none"):
            name = " ".join(w.capitalize() for w in j["name"].split()[:4])
    if not name:
        name = nz.normalize_name(text)
    if not name or len(name) < 2:
        return SlotOutcome(note="no_name")
    run.slots["patient_name"] = name
    run.scratch["name_confirmed"] = False
    return SlotOutcome(ok=True)


async def parse_age(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    age = nz.extract_age(text)
    if age is None:
        return SlotOutcome(note="no_age")
    run.slots["age"] = age
    return SlotOutcome(ok=True)


async def parse_gender(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    g = nz.extract_gender(text)
    if g is None:
        return SlotOutcome(note="no_gender")
    run.slots["gender"] = g
    return SlotOutcome(ok=True)


# ── address ─────────────────────────────────────────────────────────────────
async def parse_address(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    res = ctx.kit.resolve("area", text)
    ctx.emit("tool", {"name": "resolve_entity", "args": {"entity": "area", "text": text}, "status": res.status,
                      "result": [(c.name, c.score) for c in res.candidates[:4]]})
    out = SlotOutcome()
    run.scratch["address_text"] = text.strip()
    if res.status == "resolved" or res.status == "confirm" and res.best.score >= 0.80:
        return _set_area(ctx, run, res.best, out)
    if res.status == "confirm":
        c = res.best
        q = f"Kya aap {c.name} ke baare mein keh rahe hain?"
        run.pending = Pending("confirm_slot", {"slot": "address", "area": {"id": c.id, "name": c.name}}, q)
        out.say.append(q)
        out.asked = True
        return out
    if res.status == "ambiguous":
        labels = _names([{"name": c.name} for c in res.candidates])
        q = f"Is naam ke kai areas hain: {labels}. Kaunsa wala hai?"
        run.pending = Pending("choose_area", {"cands": [{"id": c.id, "name": c.name} for c in res.candidates]}, q)
        out.say.append(q)
        out.asked = True
        return out
    out.note = "no_area"
    return out


def _set_area(ctx: TurnCtx, run: WorkflowRun, cand, out: SlotOutcome) -> SlotOutcome:
    rec = cand.record if hasattr(cand, "record") else ctx.kit.get("area", cand["id"])
    name = cand.name if hasattr(cand, "name") else cand["name"]
    aid = cand.id if hasattr(cand, "id") else cand["id"]
    covered = (rec.get("covered", "yes").lower() == "yes")
    if not covered:
        run.slots.pop("address", None)
        run.scratch["address_attempts"] = run.scratch.get("address_attempts", 0) + 1
        out.say.append(f"Maaf kijiye, abhi {name} mein home sampling ki service maujood nahi hai. "
                       "Kya aap kisi aur address par sample dena chahenge?")
        out.asked = False
        out.note = "not_covered"
        return out
    run.slots["address"] = {"text": run.scratch.get("address_text", name), "area_id": aid, "area": name,
                            "city": rec.get("city", ""), "branch_id": rec.get("branch_id", ""),
                            "home_fee": rec.get("home_fee", "")}
    has_number = bool(nz.find_numbers(run.scratch.get("address_text", "")) or
                      re.search(r"\d", run.scratch.get("address_text", "")))
    if not has_number and not run.scratch.get("house_asked"):
        run.scratch["house_asked"] = True
        q = "Ghar ya flat ka number aur street bhi bata dein."
        run.pending = Pending("house", {}, q)
        out.say.append(q)
        out.asked = True
        return out
    out.ok = True
    return out


async def resolve_area_pending(ctx: TurnCtx, session: Session, run: WorkflowRun, text: str) -> SlotOutcome | None:
    p = run.pending
    if p is None or p.kind not in ("confirm_slot", "choose_area", "house") or \
            (p.kind == "confirm_slot" and p.data.get("slot") != "address"):
        return None
    out = SlotOutcome()
    if p.kind == "house":
        run.pending = None
        run.scratch["address_text"] = (run.scratch.get("address_text", "") + " " + text).strip()
        run.slots["address"]["text"] = run.scratch["address_text"]
        out.ok = True
        return out
    if p.kind == "confirm_slot":
        yn = nz.yes_no(text)
        if yn == "yes":
            run.pending = None
            return _set_area(ctx, run, p.data["area"], out)
        if yn == "no":
            run.pending = None
            out.say.append("Theek hai, address dobara bata dein, area ka naam ke saath.")
            return out
        run.pending = None
        return await parse_address(text, ctx, session, run)
    cands = p.data["cands"]
    idx = _ordinal_index(text, len(cands))
    if idx is None:
        idx = _match_label(ctx, "area", text, cands)
    if idx is not None:
        run.pending = None
        return _set_area(ctx, run, cands[idx], out)
    run.pending = None
    return await parse_address(text, ctx, session, run)


# ── date / time ─────────────────────────────────────────────────────────────
async def parse_datetime(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    sel = nz.resolve_date(text, ctx.today)
    part = sel.part or run.scratch.get("date_part")
    day = sel.day or (date.fromisoformat(run.scratch["date_day"]) if run.scratch.get("date_day") else None)
    out = SlotOutcome()
    if day is None and part is None and sel.hour is None:
        out.note = "no_date"
        return out
    if day is not None and day < ctx.today:
        run.scratch.pop("date_day", None)
        out.say.append("Yeh din guzar chuka hai. Aane wala koi din bata dein.")
        out.note = "past_date"
        return out
    if day is None:
        run.scratch["date_part"] = part
        out.note = "need_day"
        return out
    if part is None:
        run.scratch["date_day"] = day.isoformat()
        out.note = "need_part"
        return out
    if not ctx.backend.slot_available(day, part):
        alts = [s["part"] for s in ctx.backend.get_slots(day).data.get("slots", [])]
        run.scratch["date_day"] = day.isoformat()
        run.scratch.pop("date_part", None)
        alt_txt = _names([{"name": nz.part_label(a)} for a in alts]) if alts else ""
        out.say.append(f"Maaf kijiye, us waqt slot khali nahi hai." + (f" {alt_txt} ka slot mil sakta hai." if alt_txt else ""))
        out.note = "slot_unavailable"
        return out
    run.slots["date"] = {"day": day.isoformat(), "part": part, "hour": sel.hour, "label": sel.label}
    run.scratch.pop("date_day", None)
    run.scratch.pop("date_part", None)
    ctx.emit("tool", {"name": "get_slots", "args": {"date": day.isoformat()}, "result": part})
    out.ok = True
    return out


def spoken_when(slot: dict, today: date) -> str:
    d = date.fromisoformat(slot["day"])
    delta = (d - today).days
    names = {0: "aaj", 1: "kal", 2: "parson"}
    day_txt = names.get(delta, d.strftime("%d %B"))
    return f"{day_txt} {nz.part_label(slot['part'])}"


# ── phone ───────────────────────────────────────────────────────────────────
async def parse_phone(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    digits = nz.extract_digits(text)
    if not digits:
        return SlotOutcome(note="no_digits")
    buf = run.scratch.get("phone_buf", "") + digits
    res = nz.normalize_phone(buf)
    if res.valid:
        run.scratch.pop("phone_buf", None)
        run.scratch["phone_candidate"] = res.digits
        q = f"Aap ka number {nz.speak_digits(res.digits)} hai. Theek hai?"
        run.pending = Pending("confirm_slot", {"slot": "phone", "value": res.digits}, q)
        return SlotOutcome(say=[q], asked=True)
    if res.reason == "short":
        run.scratch["phone_buf"] = buf
        return SlotOutcome(note="phone_partial")
    run.scratch.pop("phone_buf", None)
    return SlotOutcome(say=["Number sahi nahi lag raha. Poora number dobara bata dein."], note="phone_invalid")


async def resolve_phone_pending(ctx: TurnCtx, session: Session, run: WorkflowRun, text: str) -> SlotOutcome | None:
    p = run.pending
    if p is None or p.kind != "confirm_slot" or p.data.get("slot") != "phone":
        return None
    yn = nz.yes_no(text)
    if yn == "yes":
        run.slots["phone"] = p.data["value"]
        session.facts["phone"] = p.data["value"]
        run.pending = None
        return SlotOutcome(ok=True)
    if yn == "no":
        run.pending = None
        run.scratch.pop("phone_buf", None)
        return SlotOutcome(say=["Theek hai, number dobara bata dein."])
    run.pending = None
    return await parse_phone(text, ctx, session, run)


# ── booking reference / action ──────────────────────────────────────────────
async def parse_booking_ref(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    m = re.search(r"h\s*s\s*[-\s]?(\d{4})", text.lower())
    digits = m.group(1) if m else None
    if not digits:
        d = nz.extract_digits(text)
        if len(d) == 4:
            digits = d
        elif len(d) > 4 and d.endswith(d[-4:]) and len(d) <= 6:
            digits = d[-4:]
    if not digits:
        return SlotOutcome(note="no_ref")
    run.slots["booking_ref"] = f"HS-{digits}"
    return SlotOutcome(ok=True)


_ACTIONS = {
    "reschedule": {"badal", "badalna", "badalni", "badlo", "reschedule", "shift", "change", "aage", "peeche", "doosre",
                   "dusre", "alag", "waqt", "taareekh", "tareekh", "din"},
    "cancel": {"cancel", "khatam", "mansookh", "nahi", "chahiye", "hata", "hatao", "raddi"},
    "status": {"status", "kahan", "kab", "kya", "hua", "halat", "pata", "maloom", "check", "dekhna", "batao", "bata"},
}


async def parse_action(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    toks = set(nz.tokens(text))
    scores = {a: len(toks & w) for a, w in _ACTIONS.items()}
    if "cancel" in toks:
        scores["cancel"] += 3
    if toks & {"reschedule", "badal", "badalna", "badalni"}:
        scores["reschedule"] += 3
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return SlotOutcome(note="no_action")
    if sorted(scores.values())[-1] == sorted(scores.values())[-2] and scores[best] < 3:
        return SlotOutcome(note="no_action")
    run.slots["action"] = best
    return SlotOutcome(ok=True)


# ── generic free text ───────────────────────────────────────────────────────
async def parse_text(text: str, ctx: TurnCtx, session: Session, run: WorkflowRun) -> SlotOutcome:
    t = text.strip()
    return SlotOutcome(ok=bool(t))


PARSERS = {
    "tests": parse_tests, "name": parse_name, "age": parse_age, "gender": parse_gender, "address": parse_address,
    "datetime": parse_datetime, "phone": parse_phone, "booking_ref": parse_booking_ref, "action": parse_action,
    "text": parse_text,
}

# which slot name each type fills (for opportunistic filling)
SLOT_OF_TYPE = {"tests": "tests", "name": "patient_name", "age": "age", "gender": "gender", "address": "address",
                "datetime": "date", "phone": "phone", "booking_ref": "booking_ref", "action": "action"}
