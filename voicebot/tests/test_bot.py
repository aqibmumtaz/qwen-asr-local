from datetime import date

import pytest

from app.backend.mock_api import MockBackend
from app.bot import Bot
from app.config import Settings

TODAY = date(2026, 10, 21)


def make_bot(**kw):
    s = Settings(llm_base_url="")
    return Bot(settings=s, routers={}, today=TODAY, **kw)


async def talk(bot, session, lines):
    out = []
    for l in lines:
        r = await bot.turn(session, l)
        out.append((l, r))
    return out


def show(turns):
    return "\n".join(f"U: {u}\nB: {r.text}   [{r.route['decision']}/{r.route.get('workflow')} {r.route['source']}]" for u, r in turns)


async def test_greeting_then_start_home_sampling():
    bot = make_bot()
    s = bot.new_session()
    g = bot.greeting(s)
    assert "home sampling" in g.text
    t = await talk(bot, s, ["mujhe ghar par sample dena hai"])
    print(show(t))
    assert t[0][1].route["decision"] == "switch" and t[0][1].route["workflow"] == "home_sampling"
    assert s.active and s.active.name == "home_sampling"
    assert "Kaun se tests" in t[0][1].text


async def test_full_booking_through_bot():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "ali raza", "painteesh saal", "mard",
                            "dha karachi house 12", "kal subah", "0300 1234567", "haan", "haan"])
    print(show(t))
    assert "H S" in t[-1][1].text
    assert s.active is None


async def test_faq_midflow_returns_to_question():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "lab kitne baje khulti hai"])
    print(show(t))
    r = t[1][1]
    assert r.route["decision"] == "faq"
    assert "khuli" in r.text and "naam" in r.text      # answer + the pending question again
    assert s.active.name == "home_sampling"


async def test_lookup_ambiguous_then_price_midflow():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "vitamin d ki price kya hai", "total wala"])
    print(show(t))
    assert t[1][1].route["decision"] == "lookup" and "options" in t[1][1].text
    assert "ki price" in t[2][1].text and "char hazaar paanch sau" in t[2][1].text   # 4500
    assert "naam" in t[2][1].text                                           # returned to the pending question


async def test_lookup_single_test_without_workflow():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ki price kya hai"])
    print(show(t))
    assert "aath sau" in t[0][1].text


async def test_switch_with_confirmation_and_resume():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "ali raza", "pehle meri booking ka status batao"])
    print(show(t))
    assert t[2][1].route["decision"] == "switch"
    assert s.pending and s.pending.kind == "switch"            # progress exists -> confirm first
    t2 = await talk(bot, s, ["haan", "hs 1000", "0300 1234567", "haan", "status"])
    print(show(t2))
    assert "confirm hai" in t2[-1][1].text
    assert s.pending and s.pending.kind == "resume"             # offers to go back
    t3 = await talk(bot, s, ["haan"])
    print(show(t3))
    assert s.active.name == "home_sampling" and s.active.slots["patient_name"] == "Ali Raza"
    assert "umar" in t3[0][1].text


async def test_switch_declined_keeps_workflow():
    bot = make_bot()
    s = bot.new_session()
    await talk(bot, s, ["cbc ka test ghar par karwana hai", "ali raza", "meri booking cancel karni hai"])
    assert s.pending.kind == "switch"
    t = await talk(bot, s, ["nahi"])
    print(show(t))
    assert s.active.name == "home_sampling" and not s.suspended
    assert "umar" in t[0][1].text


async def test_no_switch_during_readback():
    bot = make_bot()
    s = bot.new_session()
    lines = ["cbc ka test ghar par karwana hai", "ali raza", "painteesh saal", "mard", "dha karachi house 12", "kal subah",
             "0300 1234567", "haan"]
    await talk(bot, s, lines)
    assert s.active.pending.kind == "confirm_step"
    t = await talk(bot, s, ["meri booking ka status batao"])
    print(show(t))
    assert s.active.name == "home_sampling"
    assert "mukammal" in t[0][1].text


async def test_human_request_transfers():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "insaan se baat karwao"])
    assert t[1][1].transfer == "caller_requested" and s.ended


async def test_goodbye_after_done():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["allah hafiz"])
    assert s.ended and "Allah hafiz" in t[0][1].text


async def test_goodbye_midflow_asks_first():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["cbc ka test ghar par karwana hai", "ali raza", "khuda hafiz"])
    print(show(t))
    assert not s.ended and s.pending.kind == "end_confirm"


async def test_unclear_three_times_transfers():
    bot = make_bot()
    s = bot.new_session()
    t = await talk(bot, s, ["xyzzy", "qwerty", "asdf"])
    print(show(t))
    assert t[-1][1].transfer == "unclear"


async def test_second_switch_depth_limit():
    bot = make_bot()
    s = bot.new_session()
    s.suspended = [bot.workflows["home_sampling"].new_run(), bot.workflows["home_sampling"].new_run()]
    await talk(bot, s, ["cbc ka test ghar par karwana hai"])
    r = await bot.turn(s, "meri booking ka status batao")
    print(r.text, r.route)
    assert r.transfer == "too_many_switches" or s.pending
