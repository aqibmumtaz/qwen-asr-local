"""Code-only parts of the booking_status workflow: verification, status, reschedule, cancel."""
from datetime import date

from app.flow import slots as sl
from app.flow.form import HookResult
from app.nlp import normalizers as nz

_STATUS_HI = {"confirmed": "confirm hai", "cancelled": "cancel ho chuki hai"}


def _when(b: dict, today: date) -> str:
    return sl.spoken_when({"day": b["date"], "part": b["part"]}, today)


async def verify(ctx, session, run):
    ref = run.slots.get("booking_ref", "")
    phone = run.slots.get("phone", "")
    r = ctx.backend.get_booking(ref)
    ctx.emit("tool", {"name": "get_booking", "args": {"ref": ref}, "ok": r.ok})
    if not r.ok and r.error not in ("not_found",):
        return HookResult(say=["Maaf kijiye, system mein masla hai. Main aap ko hamare saathi se mila deti hoon."],
                          ok=False, transfer="backend_error")
    if r.ok and r.data.get("phone") == phone:
        run.scratch["booking"] = r.data
        session.facts["phone"] = phone
        return HookResult()
    fails = run.scratch.get("verify_fail", 0) + 1
    run.scratch["verify_fail"] = fails
    if fails >= 2:
        return HookResult(say=["Maaf kijiye, details match nahi ho rahin. Main aap ko hamare saathi se mila deti hoon."],
                          ok=False, transfer="verification_failed")
    return HookResult(say=["Reference number aur phone number match nahi hue. Dobara koshish karte hain."],
                      ok=False, clear=["booking_ref", "phone"])


async def report_status(ctx, session, run):
    b = run.scratch["booking"]
    b = ctx.backend.get_booking(b["ref"]).data or b
    tests = ", ".join(t if isinstance(t, str) else t.get("name", "") for t in b.get("tests", []))
    if b["status"] == "cancelled":
        return HookResult(say=[f"Aap ki booking {b['ref'].replace('-', ' ')} cancel ho chuki hai."], done=True)
    return HookResult(say=[f"Aap ki booking {_when(b, ctx.today)} ke liye {_STATUS_HI.get(b['status'], b['status'])}. "
                           f"Tests: {tests}."], done=True)


def render_vars(ctx, session, run):
    b = run.scratch.get("booking")
    if not b:
        return {}
    act = run.slots.get("action")
    if act == "cancel":
        txt = f"Aap booking {b['ref'].replace('-', ' ')} cancel karna chahte hain. Theek hai?"
    elif act == "reschedule" and run.slots.get("date"):
        txt = (f"Aap booking {b['ref'].replace('-', ' ')} ko {sl.spoken_when(run.slots['date'], ctx.today)} par "
               f"le jana chahte hain. Theek hai?")
    else:
        txt = ""
    return {"confirm_text": txt}


async def execute(ctx, session, run):
    b = run.scratch["booking"]
    act = run.slots["action"]
    if ctx.settings.shadow_mode:
        return HookResult(say=["Aap ki request note kar li gayi hai. Hamara saathi aap se rabta karega."], done=True)
    if act == "cancel":
        r = ctx.backend.cancel(b["ref"])
        ctx.emit("tool", {"name": "cancel_booking", "args": {"ref": b["ref"]}, "ok": r.ok})
        if not r.ok:
            return HookResult(say=["Maaf kijiye, booking cancel nahi ho saki. Main aap ko hamare saathi se mila deti hoon."],
                              ok=False, transfer="backend_error")
        return HookResult(say=["Aap ki booking cancel kar di gayi hai. Shukriya."], done=True)
    d = run.slots["date"]
    r = ctx.backend.reschedule(b["ref"], date.fromisoformat(d["day"]), d["part"])
    ctx.emit("tool", {"name": "reschedule_booking", "args": {"ref": b["ref"], "date": d}, "ok": r.ok})
    if not r.ok and r.error == "slot_unavailable":
        return HookResult(say=["Maaf kijiye, us waqt slot khali nahi hai. Koi aur waqt bata dein."],
                          ok=False, clear=["date"])
    if not r.ok:
        return HookResult(say=["Maaf kijiye, waqt tabdeel nahi ho saka. Main aap ko hamare saathi se mila deti hoon."],
                          ok=False, transfer="backend_error")
    return HookResult(say=[f"Aap ki booking ka waqt {sl.spoken_when(d, ctx.today)} kar diya gaya hai. Shukriya."], done=True)
