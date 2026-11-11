"""Text-to-speech client for the OmniVoice websocket (OMNI_TTS).

The exact wire format is not documented here, so the client accepts the common shapes and
`scripts/ping_endpoints.py` prints what the server actually sends:
  - JSON request {"text": "..."}  (field name from TTS_TEXT_FIELD)
  - responses: binary PCM16 frames, or JSON events carrying base64 audio ("audio"/"data"/"chunk"),
    ending on a JSON {"type": "done"/"end"/"complete"} event or when the server closes.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator

from app.config import Settings, get_settings


@dataclass
class TtsStats:
    first_audio_ms: int = 0
    total_ms: int = 0
    bytes: int = 0
    error: str = ""
    messages: list[str] = None   # first few JSON messages, for protocol discovery


class WsTts:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.last = TtsStats(messages=[])

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks. Cancel the consuming task to stop playback (barge-in)."""
        import websockets
        st = self.last = TtsStats(messages=[])
        t0 = time.perf_counter()
        hdr = {"Authorization": f"Bearer {self.s.tts_token}"} if self.s.tts_token else {}
        try:
            async with websockets.connect(self.s.tts_url, open_timeout=20, max_size=None, additional_headers=hdr) as ws:
                req = {self.s.tts_text_field: text}
                if voice or self.s.__dict__.get("tts_voice"):
                    req["voice"] = voice
                await ws.send(json.dumps(req))
                while True:
                    try:
                        m = await asyncio.wait_for(ws.recv(), timeout=self.s.tts_timeout)
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break
                    if isinstance(m, (bytes, bytearray)):
                        if not st.first_audio_ms:
                            st.first_audio_ms = int((time.perf_counter() - t0) * 1000)
                        st.bytes += len(m)
                        yield bytes(m)
                        continue
                    if len(st.messages) < 6:
                        st.messages.append(m[:200])
                    try:
                        d = json.loads(m)
                    except json.JSONDecodeError:
                        continue
                    typ = str(d.get("type", "")).lower()
                    b64 = d.get("audio") or d.get("data") or d.get("chunk") or d.get("delta")
                    if isinstance(b64, str) and b64:
                        pcm = base64.b64decode(b64)
                        if not st.first_audio_ms:
                            st.first_audio_ms = int((time.perf_counter() - t0) * 1000)
                        st.bytes += len(pcm)
                        yield pcm
                    if typ in ("done", "end", "complete", "completed", "finished", "audio.done", "response.done") or d.get("done"):
                        break
                    if typ == "error" or d.get("error"):
                        st.error = str(d.get("error") or d)[:200]
                        break
        except Exception as e:
            st.error = f"{type(e).__name__}: {e}"[:200]
        finally:
            st.total_ms = int((time.perf_counter() - t0) * 1000)

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        out = bytearray()
        async for ch in self.stream(text, voice):
            out += ch
        return bytes(out)
