"""STT/TTS clients against local fake servers (the real Spaces are private until tokens are provided)."""
import asyncio
import base64
import json
import threading
import time

import pytest
import uvicorn
import websockets
from fastapi import FastAPI, File, Header, UploadFile

from app.config import Settings
from app.services.stt import HttpStt, WsStt, pcm16_to_wav
from app.services.tts import WsTts


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def stt_server():
    seen = {}
    app = FastAPI()

    @app.post("/chughtai/v1/audio/transcriptions")
    async def tr(file: UploadFile = File(...), authorization: str | None = Header(None)):
        data = await file.read()
        seen["auth"], seen["bytes"] = authorization, len(data)
        return {"text": "यह एक टेस्ट है", "raw_text": "यह एक टेस्ट है", "romanized_text": "ye ek test hai"}

    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}", seen
    srv.should_exit = True


async def test_http_stt_returns_roman_text_and_sends_token(stt_server):
    url, seen = stt_server
    s = Settings(stt_http_url=url, stt_variant="chughtai", stt_token="tok123", llm_base_url="")
    r = await HttpStt(s).transcribe_pcm(b"\x00\x01" * 8000, 16000)
    assert r.error == "" and r.text == "ye ek test hai"
    assert seen["auth"] == "Bearer tok123" and seen["bytes"] > 16000


async def test_http_stt_reports_errors_instead_of_raising():
    s = Settings(stt_http_url="http://127.0.0.1:9", stt_variant="chughtai", llm_base_url="", stt_timeout=2)
    r = await HttpStt(s).transcribe_pcm(b"\x00\x00" * 100)
    assert r.text == "" and r.error


def test_wav_header():
    w = pcm16_to_wav(b"\x00\x00" * 100, 16000)
    assert w[:4] == b"RIFF" and b"WAVE" in w[:12]


async def _serve_ws(handler):
    port = _free_port()
    server = await websockets.serve(handler, "127.0.0.1", port)
    return server, f"ws://127.0.0.1:{port}"


async def test_ws_stt_collects_completed_segments():
    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created"}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "response.create":
                await ws.send(json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                                          "transcript": "x", "romanized_text": "pehla hissa"}))
                await ws.send(json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                                          "transcript": "y", "romanized_text": "doosra hissa"}))
                await asyncio.sleep(0.05)
                await ws.close()
                return

    server, url = await _serve_ws(handler)
    try:
        s = Settings(stt_url=url, llm_base_url="")
        r = await WsStt(s).transcribe_pcm(b"\x00\x01" * 4800, 24000)
        assert r.text == "pehla hissa doosra hissa", r
    finally:
        server.close()


async def test_ws_tts_binary_frames_and_cancellation():
    async def handler(ws):
        req = json.loads(await ws.recv())
        assert req["text"] == "salam"
        for _ in range(5):
            await ws.send(b"\x01\x00" * 100)
        await ws.close()

    server, url = await _serve_ws(handler)
    try:
        s = Settings(tts_url=url, llm_base_url="")
        tts = WsTts(s)
        audio = await tts.synthesize("salam")
        assert len(audio) == 1000 and tts.last.first_audio_ms >= 0 and not tts.last.error
        # barge-in: stop consuming after the first chunk
        got = []
        async for ch in tts.stream("salam"):
            got.append(ch)
            break
        assert len(got) == 1
    finally:
        server.close()


async def test_ws_tts_json_base64_events():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "audio", "audio": base64.b64encode(b"\x02\x00" * 50).decode()}))
        await ws.send(json.dumps({"type": "done"}))

    server, url = await _serve_ws(handler)
    try:
        tts = WsTts(Settings(tts_url=url, llm_base_url=""))
        assert len(await tts.synthesize("x")) == 100
    finally:
        server.close()


async def test_ws_tts_error_is_reported():
    tts = WsTts(Settings(tts_url="ws://127.0.0.1:9", llm_base_url="", tts_timeout=1))
    audio = await tts.synthesize("x")
    assert audio == b"" and tts.last.error
