from datetime import date

from tests.helpers import load, make_ctx, run_flow


def show(tr):
    return "\n".join(f"{w:4}: {t}" for w, t in tr)


async def test_status_flow():
    ctx, wf = make_ctx(), load("booking_status")
    # seeded booking HS-1000 with phone 03001234567
    lines = ["meri booking ka status batao", "h s ek zero zero zero", "0300 1234567", "haan", "status"]
    session, run, tr, out = await run_flow(wf, ctx, lines)
    print(show(tr))
    assert out.done and "confirm hai" in tr[-1][1]


async def test_verification_failure_twice_transfers():
    ctx, wf = make_ctx(), load("booking_status")
    lines = ["status", "hs 1000", "0300 0000000", "haan", "hs 1000", "0311 1111111", "haan"]
    session, run, tr, out = await run_flow(wf, ctx, lines)
    print(show(tr))
    assert out.transfer == "verification_failed"


async def test_cancel_requires_confirm():
    ctx, wf = make_ctx(), load("booking_status")
    lines = ["booking cancel karni hai", "hs 1000", "0300 1234567", "haan", "cancel", "haan"]
    session, run, tr, out = await run_flow(wf, ctx, lines)
    print(show(tr))
    assert out.done
    assert ctx.backend.bookings["HS-1000"]["status"] == "cancelled"


async def test_reschedule():
    ctx, wf = make_ctx(), load("booking_status")
    lines = ["waqt badalna hai", "hs 1000", "0300 1234567", "haan", "waqt badalna hai", "parson shaam", "haan"]
    session, run, tr, out = await run_flow(wf, ctx, lines)
    print(show(tr))
    assert out.done
    b = ctx.backend.bookings["HS-1000"]
    assert b["date"] == "2026-10-23" and b["part"] == "evening"


async def test_reschedule_unavailable_slot_asks_again():
    ctx, wf = make_ctx(), load("booking_status")
    ctx.backend.blocked_slots.add(("2026-10-23", "evening"))
    lines = ["waqt badalna hai", "hs 1000", "0300 1234567", "haan", "waqt badalna hai", "parson shaam"]
    session, run, tr, out = await run_flow(wf, ctx, lines)
    print(show(tr))
    assert "khali nahi" in tr[-1][1]
