#!/usr/bin/env python3
"""Routing eval + bake-off on eval/routing/routing_set.yaml.

Compares, on the same labelled Roman Urdu utterances and conversation contexts:
  rules            cheap pass + rule baseline (no model)
  laya_raw         Laya zero-shot on every utterance the cheap pass does not decide (no confidence threshold)
  laya_chain       cheap pass -> Laya (>= LAYA_CONFIDENCE_MIN) -> rules
  qwen_chain       cheap pass -> Qwen -> rules (only when LLM_BASE_URL is set)
Reports accuracy, per-label recall, false switches, latency. Usage: python scripts/run_router_bakeoff.py [--no-laya] [-v]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bot import Bot  # noqa: E402
from app.config import ROOT, get_settings  # noqa: E402
from app.flow.types import Pending  # noqa: E402
from app.nlp import normalizers as nz  # noqa: E402
from app.router.chain import ChainRouter  # noqa: E402
from app.router.llm_router import QwenRouter  # noqa: E402
from app.router.rules import cheap_pass  # noqa: E402


def make_session(bot: Bot, name: str):
    s = bot.new_session(today=date(2026, 10, 21))
    A, B = bot.workflows["home_sampling"], bot.workflows["booking_status"]

    def run_of(wf, step, slots=None, pending=None, last=""):
        r = wf.new_run()
        r.step, r.slots = step, dict(slots or {})
        if pending:
            r.pending = Pending(*pending)
        s.active, s.last_bot = r, last
        return r

    cbc = [{"id": "T001", "name": "Complete Blood Count"}]
    if name == "none":
        s.last_bot = "Aap home sampling booking karna chahte hain, ya kisi booking ka status dekhna chahte hain?"
    elif name == "A_tests":
        run_of(A, "tests", {}, last="Kaun se tests karwane hain?")
    elif name == "A_name":
        run_of(A, "patient_name", {"tests": cbc}, last="Mareez ka naam kya hai?")
    elif name == "A_age":
        run_of(A, "age", {"tests": cbc, "patient_name": "Ali"}, last="Mareez ki umar kya hai?")
    elif name == "A_gender":
        run_of(A, "gender", {"tests": cbc, "patient_name": "Ali", "age": 35}, last="Mareez mard hain ya aurat?")
    elif name == "A_address":
        run_of(A, "address", {"tests": cbc, "patient_name": "Ali", "age": 35, "gender": "male"},
               last="Sample kis address par lena hai? Area ka naam aur ghar ka number bata dein.")
    elif name == "A_date":
        run_of(A, "date", {"tests": cbc, "patient_name": "Ali", "age": 35, "gender": "male", "address": {"area": "DHA", "text": "x", "area_id": "A004"}},
               last="Kis din aur kis waqt aana theek rahega?")
    elif name == "A_phone":
        run_of(A, "phone", {"tests": cbc, "patient_name": "Ali", "age": 35, "gender": "male", "address": {"area": "DHA", "text": "x", "area_id": "A004"},
                            "date": {"day": "2026-10-22", "part": "morning"}}, last="Rabta number bata dein.")
    elif name == "A_confirm":
        q = "Confirm kar leti hoon: Ali, 35 saal, Complete Blood Count, DHA, kal subah. Sab theek hai?"
        run_of(A, "confirm", {"tests": cbc, "patient_name": "Ali", "age": 35}, pending=("confirm_step", {"step": "confirm"}, q), last=q)
    elif name == "B_ref":
        run_of(B, "ref", {}, last="Booking ka reference number bata dein, jaise H S ek zero zero ek.")
    elif name == "B_action":
        run_of(B, "action", {"booking_ref": "HS-1000", "phone": "03001234567"},
               last="Aap booking ka status dekhna chahte hain, waqt badalna chahte hain, ya cancel karna chahte hain?")
    elif name == "paused":
        r = A.new_run()
        r.slots, r.step, r.status = {"tests": cbc, "patient_name": "Ali"}, "age", "suspended"
        s.suspended.append(r)
        s.last_bot = "Kya main aur kisi cheez mein madad kar sakti hoon?"
    else:
        raise ValueError(name)
    return s


def evaluate(items, decisions):
    ok = 0
    conf = Counter()
    by = defaultdict(lambda: [0, 0])
    false_switch = 0
    bad = []
    for it, d in zip(items, decisions):
        want = it["label"]
        good = d.decision == want and (want != "switch" or d.workflow == it.get("workflow"))
        ok += good
        conf[(want, d.decision)] += 1
        by[want][1] += 1
        by[want][0] += good
        if d.decision == "switch" and want != "switch":
            false_switch += 1
        if not good:
            bad.append((it, d))
    return ok, by, false_switch, bad, conf


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    items = yaml.safe_load((ROOT / "eval" / "routing" / "routing_set.yaml").read_text(encoding="utf-8"))
    bot = Bot(settings=s, routers={})
    # build route contexts once
    rcs = []
    for it in items:
        sess = make_session(bot, it["ctx"])
        ctx = bot._ctx([], sess)
        rcs.append(bot._route_context(ctx, sess, it["utt"], nz.clean_text(it["utt"])))
    cheap = [cheap_pass(rc) for rc in rcs]
    print(f"{len(items)} labelled utterances; cheap pass decides {sum(c is not None for c in cheap)} of them\n")

    results: dict[str, list] = {}
    rules_chain = ChainRouter(s, {})
    t0 = time.perf_counter()
    results["rules"] = [await rules_chain.route(rc) for rc in rcs]
    lat = {"rules": (time.perf_counter() - t0) / len(rcs) * 1000}

    if not a.no_laya:
        try:
            from app.router.laya_router import LayaDecisionRouter
            laya = LayaDecisionRouter(s)
            laya.warm()
            raw, t0 = [], time.perf_counter()
            n_model = 0
            for rc, c in zip(rcs, cheap):
                if c is not None:
                    raw.append(c)
                    continue
                n_model += 1
                d = await laya.route(rc)
                raw.append(d)
            lat["laya_raw"] = (time.perf_counter() - t0) / max(n_model, 1) * 1000
            results["laya_raw"] = [d if d else cheap_fallback for d, cheap_fallback in
                                   zip(raw, results["rules"])]
            chain = ChainRouter(s, {"laya": laya})
            s.router_primary, s.router_fallback = "laya", "none"
            t0 = time.perf_counter()
            results["laya_chain"] = [await chain.route(rc) for rc in rcs]
            lat["laya_chain"] = (time.perf_counter() - t0) / len(rcs) * 1000
            print("laya conf threshold:", s.laya_conf_min, "| chain stats:", chain.stats)
        except Exception as e:
            print("Laya unavailable:", type(e).__name__, str(e)[:200])
    if bot.llm.enabled:
        qchain = ChainRouter(s, {"qwen": QwenRouter(bot.llm, s.qwen_router_conf_min)})
        s.router_primary, s.router_fallback = "qwen", "none"
        results["qwen_chain"] = [await qchain.route(rc) for rc in rcs]
    else:
        print("Qwen router not evaluated: LLM_BASE_URL is not set.")

    print("\n%-12s %8s %12s %8s" % ("router", "accuracy", "false-switch", "ms/utt"))
    summary = {}
    for name, decs in results.items():
        ok, by, fs, bad, conf = evaluate(items, decs)
        summary[name] = {"accuracy": round(ok / len(items), 3), "false_switch": fs, "ms": round(lat.get(name, 0), 1),
                         "per_label": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())}}
        print("%-12s %7.1f%% %12d %8.0f" % (name, 100 * ok / len(items), fs, lat.get(name, 0)))
        print("   per label:", ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items())))
        if a.v:
            for it, d in bad:
                print(f"     [{it['ctx']}] {it['utt']!r}: want {it['label']}{'/' + it.get('workflow', '') if it.get('workflow') else ''}, "
                      f"got {d.decision}{'/' + str(d.workflow) if d.workflow else ''} ({d.source} {d.confidence:.2f})")
    (ROOT / "eval" / "routing" / "results.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
