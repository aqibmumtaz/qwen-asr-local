"""LiveKit plugin wrappers against local fake STT/TTS servers."""
import json
import threading
import time

import pytest

pytest.importorskip("livekit.agents")
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from livekit import rtc  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.livekit_plugins import BotSTT, BotTTS  # noqa: E402
from tests.test_services import _free_port  # noqa: E402


@pytest.fixture(scope="module")
def stt_url():
    app = FastAPI()

    @app.post("/chughtai/v1/audio/transcriptions")
    async def tr(file: UploadFile = File(...)):
        await file.read()
        return {"text": "x", "romanized_text": "kal subah"}

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True


async def test_bot_stt_recognizes_audio_frame(stt_url):
    s = Settings(stt_http_url=stt_url, stt_variant="chughtai", llm_base_url="")
    stt_ = BotSTT(s)
    frame = rtc.AudioFrame(data=b"\x00\x01" * 16000, sample_rate=16000, num_channels=1, samples_per_channel=16000)
    ev = await stt_.recognize([frame])
    assert ev.alternatives[0].text == "kal subah"


async def test_bot_tts_streams_audio_frames():
    async def handler(ws):
        await ws.recv()
        for _ in range(4):
            await ws.send(b"\x01\x00" * 2400)
        await ws.close()

    port = _free_port()
    server = await websockets.serve(handler, "127.0.0.1", port)
    try:
        s = Settings(tts_url=f"ws://127.0.0.1:{port}", llm_base_url="")
        t = BotTTS(s)
        n = 0
        async with t.synthesize("assalam o alaikum") as stream:
            async for ev in stream:
                n += len(ev.frame.data)
        assert n >= 9600     # all PCM arrived (frame data counts int16 samples)
    finally:
        server.close()
