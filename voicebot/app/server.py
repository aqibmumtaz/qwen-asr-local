"""FastAPI server: test console (chat now, voice when LiveKit is configured), scenario runner, config view.

Run:  .venv/bin/python -m app.server     (or scripts/run_dev.sh)  ->  http://localhost:7860
"""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.bot import Bot
from app.config import ROOT, get_settings
from app.scenarios import (check_expect, check_final, load_scenarios, new_scenario_session, run_many,
                           run_scenario)

settings = get_settings()
bot = Bot(settings=settings)
app = FastAPI(title="Chughtai voice bot console")
UI = ROOT / "ui"
scenario_runs: dict[str, dict] = {}     # session id -> guided scenario progress


class NewSession(BaseModel):
    caller_phone: str | None = None
    scenario_id: str | None = None
    greet: bool = True


class Turn(BaseModel):
    session_id: str
    text: str


def _result_json(r) -> dict:
    return {"replies": r.replies, "text": r.text, "route": r.route, "events": r.events, "state": r.state,
            "timings": r.timings, "ended": r.ended, "transfer": r.transfer}


def _scenario(sc_id: str) -> dict:
    for sc in load_scenarios():
        if sc["id"] == sc_id:
            return sc
    raise HTTPException(404, f"unknown scenario {sc_id}")


@app.get("/")
async def index():
    return FileResponse(UI / "index.html")


@app.get("/api/config")
async def config():
    s = settings
    return {
        "llm_enabled": bot.llm.enabled, "llm_model": s.llm_model,
        "router": {"primary": s.router_primary, "fallback": s.router_fallback, "fallback_on": s.router_fallback_on,
                   "laya_conf_min": s.laya_conf_min, "available": list(bot.router.routers)},
        "stt_url": s.stt_url, "tts_url": s.tts_url, "livekit_url": s.livekit_url,
        "shadow_mode": s.shadow_mode, "workflows": {n: w.title for n, w in bot.workflows.items()},
        "router_stats": bot.router.stats, "backend_mode": s.booking_api_mode, "date": date.today().isoformat(),
    }


@app.post("/api/session")
async def new_session(req: NewSession):
    sc = _scenario(req.scenario_id) if req.scenario_id else None
    if sc:
        sess = new_scenario_session(bot, sc)
        scenario_runs[sess.id] = {"scenario": sc, "step": 0, "failures": [], "tools": []}
    else:
        sess = bot.new_session(req.caller_phone)
    greeting = bot.greeting(sess) if req.greet or (sc and sc.get("greet")) else None
    return {"session_id": sess.id, "greeting": greeting.text if greeting else "", "state": bot.snapshot(sess),
            "scenario": ({"id": sc["id"], "title": sc.get("title"), "steps": [s["say"] for s in sc["steps"]]} if sc else None)}


@app.post("/api/turn")
async def turn(req: Turn):
    sess = bot.sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    if sess.ended:
        raise HTTPException(409, "session ended")
    r = await bot.turn(sess, req.text)
    out = _result_json(r)
    run = scenario_runs.get(sess.id)
    if run:     # guided mode: check the caller's line against the scenario's expectation for this step
        sc, i = run["scenario"], run["step"]
        tools = [e["name"] for e in r.events if e["kind"] == "tool"]
        run["tools"] += tools
        if i < len(sc["steps"]):
            step = sc["steps"][i]
            matches = step["say"].strip().lower() == req.text.strip().lower()
            bad = check_expect(step.get("expect") or {}, r, sess, tools) if matches else []
            out["check"] = {"step": i, "scripted": matches, "ok": not bad, "failures": bad}
            if matches:
                run["step"] += 1
                run["failures"] += bad
        if run["step"] >= len(sc["steps"]) or sess.ended:
            fin = check_final(sc.get("final") or {}, sess.scratch["backend"], run["tools"])
            out["scenario_done"] = {"ok": not run["failures"] and not fin, "failures": run["failures"] + fin}
    return out


@app.post("/api/token")
async def livekit_token():
    """Mint a LiveKit access token for the browser console (voice mode). The agent worker joins the room automatically."""
    try:
        from livekit import api as lkapi
    except ImportError:
        raise HTTPException(501, "livekit-api is not installed")
    import uuid
    room = f"call-{uuid.uuid4().hex[:8]}"
    identity = f"tester-{uuid.uuid4().hex[:6]}"
    tok = (lkapi.AccessToken(settings.livekit_key, settings.livekit_secret).with_identity(identity)
           .with_grants(lkapi.VideoGrants(room_join=True, room=room)).to_jwt())
    return {"url": settings.livekit_url.replace("ws://", "ws://").replace("localhost", "localhost"), "token": tok,
            "room": room, "identity": identity}


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    sess = bot.sessions.get(sid)
    if not sess:
        raise HTTPException(404, "unknown session")
    return {"state": bot.snapshot(sess), "history": sess.history}


class BackendSwitch(BaseModel):
    session_id: str
    fail_next: str | None = None
    block_slot: list[str] | None = None


@app.post("/api/backend")
async def backend_switch(req: BackendSwitch):
    sess = bot.sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    be = sess.scratch["backend"]
    if req.fail_next is not None:
        be.fail_next = req.fail_next or None
    if req.block_slot:
        be.blocked_slots.add((req.block_slot[0], req.block_slot[1]))
    return {"fail_next": be.fail_next, "blocked": sorted(map(list, be.blocked_slots)),
            "bookings": list(be.bookings.values())}


@app.get("/api/bookings/{sid}")
async def bookings(sid: str):
    sess = bot.sessions.get(sid)
    if not sess:
        raise HTTPException(404, "unknown session")
    return list(sess.scratch["backend"].bookings.values())


@app.get("/api/scenarios")
async def scenarios():
    return [{"id": s["id"], "title": s.get("title", s["id"]), "tags": s.get("tags", []), "steps": len(s["steps"]),
             "lines": [st["say"] for st in s["steps"]], "greet": s.get("greet", False)} for s in load_scenarios()]


class RunReq(BaseModel):
    tag: str | None = None
    ids: list[str] | None = None


@app.post("/api/scenarios/run")
async def run_scenarios(req: RunReq):
    scs = load_scenarios()
    if req.ids:
        scs = [s for s in scs if s["id"] in req.ids]
    elif req.tag and req.tag != "all":
        scs = [s for s in scs if req.tag in s.get("tags", [])]
    t0 = time.perf_counter()
    res = await run_many(scs, settings=settings, routers=bot.router.routers, llm=bot.llm)
    return {"ms": int((time.perf_counter() - t0) * 1000), "passed": sum(r.ok for r in res), "total": len(res),
            "results": [r.as_dict() for r in res]}


app.mount("/static", StaticFiles(directory=UI), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=settings.ui_port, log_level="info")


if __name__ == "__main__":
    main()
