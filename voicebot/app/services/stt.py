"""Speech-to-text clients for the Qwen3-ASR Space (Roman Urdu out).

HTTP batch (default)  POST {base}/{variant}/v1/audio/transcriptions   -> {text, raw_text, romanized_text}
WebSocket realtime     {base}/{variant}/v1/realtime (OpenAI-Realtime style)

The repo's own benchmark notes report the realtime path scoring far below the HTTP path on identical audio
(~32% vs ~68%), so the bot uses HTTP per utterance (cut by VAD) unless STT_MODE=ws.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import wave
from dataclasses import dataclass

import httpx

from app.config import Settings, get_settings

_LEAK = re.compile(r"diagnostic laib|kolars? spek|miks with inglish|paikistani diagnostic|callers speak|urdoo/hindi|"
                   r"phone call .*chugtaai lab", re.I)


@dataclass
class SttResult:
    text: str              # Roman Urdu used by the bot (romanized_text when available)
    raw_text: str = ""
    ms: int = 0
    error: str = ""
    provider: str = "http"


def pcm16_to_wav(pcm: bytes, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


class HttpStt:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @property
    def url(self) -> str:
        return f"{self.s.stt_http_url.rstrip('/')}/{self.s.stt_variant}/v1/audio/transcriptions"

    async def transcribe_wav(self, wav_bytes: bytes) -> SttResult:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.s.stt_timeout) as c:
                r = await c.post(self.url, files={"file": ("utt.wav", wav_bytes, "audio/wav")},
                                 headers=_headers(self.s.stt_token))
            ms = int((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                return SttResult("", ms=ms, error=f"HTTP {r.status_code}: {r.text[:120]}")
            j = r.json()
            txt = (j.get("romanized_text") or j.get("text") or "").strip()
            if _LEAK.search(txt):
                txt = ""
            return SttResult(txt, raw_text=(j.get("raw_text") or j.get("text") or ""), ms=ms)
        except Exception as e:
            return SttResult("", ms=int((time.perf_counter() - t0) * 1000), error=f"{type(e).__name__}: {e}"[:200])

    async def transcribe_pcm(self, pcm: bytes, rate: int = 16000) -> SttResult:
        return await self.transcribe_wav(pcm16_to_wav(pcm, rate))


class WsStt:
    """OpenAI-Realtime-style websocket; sends one utterance and collects completed transcripts."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @property
    def url(self) -> str:
        return self.s.stt_url

    async def transcribe_pcm(self, pcm: bytes, rate: int = 16000) -> SttResult:
        import numpy as np
        import websockets
        t0 = time.perf_counter()
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if rate != 24000 and a.size:
            a = np.interp(np.linspace(0, 1, int(len(a) * 24000 / rate)), np.linspace(0, 1, len(a)), a)
        pcm24 = a.astype(np.int16).tobytes()
        segs: list[str] = []
        try:
            hdr = _headers(self.s.stt_token)
            async with websockets.connect(self.url, open_timeout=20, max_size=None, additional_headers=hdr) as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "session.update", "session": {"input_audio_format": "pcm16"}}))
                step = 24000
                for i in range(0, len(pcm24), step):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                              "audio": base64.b64encode(pcm24[i:i + step]).decode()}))
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(json.dumps({"type": "response.create"}))
                got = False
                while True:
                    try:
                        m = await asyncio.wait_for(ws.recv(), timeout=(6 if got else 30))
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break
                    d = json.loads(m)
                    if d.get("type") == "conversation.item.input_audio_transcription.completed":
                        t = (d.get("romanized_text") or d.get("transcript") or d.get("text") or "").strip()
                        if t and not _LEAK.search(t):
                            segs.append(t)
                        got = True
            return SttResult(" ".join(segs), ms=int((time.perf_counter() - t0) * 1000), provider="ws")
        except Exception as e:
            return SttResult("", ms=int((time.perf_counter() - t0) * 1000), error=f"{type(e).__name__}: {e}"[:200], provider="ws")


def make_stt(settings: Settings | None = None):
    s = settings or get_settings()
    return WsStt(s) if s.stt_mode == "ws" else HttpStt(s)
