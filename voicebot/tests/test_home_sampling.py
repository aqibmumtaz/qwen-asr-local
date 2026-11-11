import pytest

from tests.helpers import load, make_ctx, run_flow


def show(tr):
    return "\n".join(f"{w:4}: {t}" for w, t in tr)


async def test_happy_path_submits_booking():
    ctx, wf = make_ctx(), load("home_sampling")
    lines = ["cbc aur lipid profile", "muhammad ali", "painteesh saal", "mard", "gulshan e iqbal block 13 d ghar number 45",
             "kal subah", "0300 1234567", "haan", "haan", "haan", "haan"]
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai"] + lines)
    print(show(tr))
    assert out.done, show(tr)
    assert run.slots["patient_name"] == "Muhammad Ali"
    assert run.slots["age"] == 35 and run.slots["gender"] == "male"
    assert [t["name"] for t in run.slots["tests"]] == ["Complete Blood Count", "Lipid Profile"]
    assert run.slots["address"]["area"] == "Gulshan-e-Iqbal"
    assert run.slots["phone"] == "03001234567"
    b = list(ctx.backend.bookings.values())[-1]
    assert b["patient_name"] == "Muhammad Ali" and b["status"] == "confirmed"
    assert "H S" in tr[-1][1]


async def test_ambiguous_test_then_choice():
    ctx, wf = make_ctx(), load("home_sampling")
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai", "vitamin d", "pehla wala"])
    print(show(tr))
    names = [t["name"] for t in run.slots.get("tests", [])]
    assert len(names) == 1 and names[0].startswith("Vitamin D")
    assert any("options" in t for w, t in tr if w == "bot")


async def test_ambiguous_test_choice_by_name():
    ctx, wf = make_ctx(), load("home_sampling")
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai", "vitamin d", "total wala"])
    assert [t["name"] for t in run.slots["tests"]] == ["Vitamin D Total"]


async def test_unknown_test_then_spelled():
    ctx, wf = make_ctx(), load("home_sampling")
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai", "sivi si", "c b c"])
    print(show(tr))
    assert [t["name"] for t in run.slots["tests"]] == ["Complete Blood Count"]


async def test_uncovered_area():
    ctx, wf = make_ctx(), load("home_sampling")
    lines = ["cbc", "ali", "30 saal", "mard", "korangi mein ghar number 5"]
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai"] + lines)
    print(show(tr))
    assert "address" not in run.slots
    assert "service maujood nahi" in tr[-1][1]


async def test_two_failures_transfer():
    ctx, wf = make_ctx(), load("home_sampling")
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai", "cbc", "ali", "pata nahi", "kuch", "kuch aur", "phir"])
    print(show(tr))
    assert out.transfer == "slot_failed_age"


async def test_correction_at_readback():
    ctx, wf = make_ctx(), load("home_sampling")
    lines = ["cbc", "ali raza", "painteesh saal", "mard", "dha karachi house 12", "kal subah", "0300 1234567", "haan",
             "nahi, kal nahi parson shaam", "haan"]
    session, run, tr, out = await run_flow(wf, ctx, ["mujhe test karwana hai"] + lines)
    print(show(tr))
    assert out.done
    b = list(ctx.backend.bookings.values())[-1]
    assert b["date"]["day"] == "2026-10-23" and b["date"]["part"] == "evening"
    assert b["address"]["area"] == "DHA Karachi"


async def test_one_sentence_start_fills_tests():
    ctx, wf = make_ctx(), load("home_sampling")
    session, run, tr, out = await run_flow(wf, ctx, ["cbc ka test ghar par karwana hai"])
    assert [t["name"] for t in run.slots["tests"]] == ["Complete Blood Count"]
    assert "naam" in tr[1][1]
