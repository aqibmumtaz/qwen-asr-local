"""Rule-based routing: a high-precision cheap pass (runs first) and a full baseline (runs last).

The cheap pass answers cases that need no model: replies to a pending question, explicit requests for a
human, goodbyes, clear FAQs, explicit resume. The baseline adds keyword/example matching for switching.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from app.nlp import normalizers as nz
from app.router.interface import RouteContext, RouteDecision

_HUMAN = ("insaan se", "insan se", "bande se", "kisi se baat", "operator", "agent se", "agent", "manager",
          "representative", "customer care", "customer service", "kisi aur se", "asal admi", "real person",
          "human", "koi banda", "staff se", "baat karwa", "baat karao", "baat karwao")
_END = ("allah hafiz", "khuda hafiz", "khudahafiz", "allahhafiz", "bas shukriya", "bas itna hi", "bas itna",
        "shukriya bas", "call khatam", "phone rakh", "alvida", "bye", "good bye", "goodbye", "bas theek hai shukriya",
        "aur kuch nahi", "kuch aur nahi", "bas shukria", "nahi shukriya", "nahi bas", "nahi shukria",
        "nahi aur kuch nahi", "nahi kuch nahi", "bas ji", "bas shukriya ji")
_RESUME_WORDS = {"wapas", "wahin", "jari", "jaari", "continue", "resume", "dobara", "phir", "pehle"}
_RESUME_PHRASES = ("wahin se", "jahan chhodi", "jahan chora", "wapas chalo", "booking jari", "pehle wali jari",
                   "continue karo", "resume karo", "dobara shuru", "wapas shuru", "pehle wali booking", "pehle wali wapas",
                   "wapas aate", "wapas aa", "wapas chalte")
_PRICE = {"price", "qeemat", "kimat", "keemat", "rate", "kitne", "kitna", "kitni", "charges", "charge", "fees", "fee",
          "cost", "paise", "rupay", "rupee", "rupaye", "kharcha"}
_GREET = ("assalam", "salam", "hello", "hi ", "hey", "salaam", "aoa")
_CLEAR_YES_NO_EXPECT = {"yes_no"}


def has_phrase(text: str, phrases) -> bool:
    t = f" {text} "
    return any(f" {p} " in t or (len(p) > 6 and p in t) for p in phrases)


def cheap_pass(ctx: RouteContext) -> RouteDecision | None:
    t = ctx.text
    toks = set(t.split())
    if not t:
        return RouteDecision("unclear", confidence=1.0, reason="empty")
    if has_phrase(t, _HUMAN) and not ctx.slot_answer:
        return RouteDecision("human", confidence=0.97, reason="asked for a person")
    if has_phrase(t, _END) and ctx.expects != "yes_no" or t in ("khuda hafiz", "allah hafiz", "bye", "shukriya", "bas", "shukria"):
        return RouteDecision("end", confidence=0.95, reason="goodbye")
    # answers to a session-level question (switch / resume confirmation, lookup choice)
    if ctx.session_pending in ("switch", "resume", "lookup_choose", "lookup_confirm", "faq_confirm"):
        return RouteDecision("continue", confidence=0.95, reason=f"answer to {ctx.session_pending}")
    if ctx.suspended and (has_phrase(t, _RESUME_PHRASES) or (toks & {"wapas", "wahin"})):
        return RouteDecision("resume", confidence=0.9, reason="resume phrase")
    if ctx.active:
        yn = nz.yes_no(ctx.utterance)
        if ctx.expects == "yes_no" and yn:
            return RouteDecision("continue", confidence=1.0, reason="yes/no to pending question")
        if ctx.expects == "choice" and (ctx.slot_answer):
            return RouteDecision("continue", confidence=0.95, reason="choice answer")
        if ctx.slot_answer and ctx.faq_status != "answer":
            return RouteDecision("continue", confidence=0.95, reason="parses as the answer")
    if ctx.faq_status == "answer" and ctx.faq_score >= 0.88:
        return RouteDecision("faq", confidence=ctx.faq_score, reason="matched FAQ")
    return None


def _strong_hits(ctx: RouteContext, wf: str) -> int:
    kws = set(ctx.workflows.get(wf, {}).get("strong_keywords", []))
    toks = set(ctx.text.split())
    return len(toks & kws)


def _example_score(ctx: RouteContext, wf: str) -> float:
    best = 0.0
    for ex in ctx.workflows.get(wf, {}).get("examples", []):
        s = (0.6 * fuzz.token_sort_ratio(ctx.text, ex) + 0.4 * fuzz.token_set_ratio(ctx.text, ex)) / 100.0
        best = max(best, s)
    return best


def score_workflows(ctx: RouteContext) -> list[tuple[str, float]]:
    out = []
    for wf in ctx.workflows:
        strong = _strong_hits(ctx, wf)
        s = 0.45 * _example_score(ctx, wf) + (0.25 + 0.2 * min(strong, 2) if strong else 0.0)
        if wf == "home_sampling" and ctx.mentions_test:
            s += 0.15
        out.append((wf, min(round(s, 3), 0.98)))
    return sorted(out, key=lambda x: -x[1])


class RulesRouter:
    """Final fallback: full rules without any model."""
    name = "rules"

    async def route(self, ctx: RouteContext) -> RouteDecision | None:
        d = cheap_pass(ctx)
        if d:
            return d
        t = ctx.text
        toks = set(t.split())
        # test price / info
        if (toks & _PRICE) and (ctx.mentions_test or "test" in toks):
            return RouteDecision("lookup", confidence=0.9, reason="price question with a test")
        ranked = score_workflows(ctx)
        if ranked and ctx.faq_status == "answer" and ctx.faq_score < 0.9:
            best, bs = ranked[0]
            if best != ctx.active and bs >= 0.8 and len(toks) >= 2:
                return RouteDecision("switch", workflow=best, confidence=bs, reason="task phrase beats a weak FAQ match")
        if ctx.faq_status == "answer":
            return RouteDecision("faq", confidence=ctx.faq_score, reason="matched FAQ")
        if not ctx.active and ctx.mentions_test and not (toks & _PRICE) and "home_sampling" in ctx.workflows:
            return RouteDecision("switch", workflow="home_sampling", confidence=0.8, reason="a test name was mentioned")
        if ranked:
            best, bs = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if best != ctx.active and bs >= 0.55 and bs - second >= 0.05 and len(toks) >= 2:
                return RouteDecision("switch", workflow=best, confidence=bs, reason="workflow keywords/examples")
        if ctx.faq_status == "confirm" and (not ctx.active or ctx.faq_score >= 0.7):
            return RouteDecision("faq", confidence=ctx.faq_score, reason="FAQ needs confirmation")
        if ctx.active:
            return RouteDecision("continue", confidence=0.6, reason="default: stay in the active workflow")
        if has_phrase(t + " ", _GREET):
            return RouteDecision("unclear", confidence=0.9, reason="greeting")
        return RouteDecision("unclear", confidence=0.5, reason="nothing matched")
