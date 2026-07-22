"""
Qwen3-ASR wrapper — LOCAL or REMOTE, same interface.

  transcribe(audio, context="", language="Hindi") -> str (Hindi text)

REMOTE (default): the GPU-hosted async server (OpenAI-Realtime protocol over WebSocket).
  - `/en/v1/realtime` — returns Hindi with language=Hindi instruction + optional biasing context.
  - Fast (GPU); this is what makes a full 80-call benchmark feasible.
LOCAL: base Qwen3-ASR via the qwen_asr package (CPU, slow; for offline use).

  from acoustic_contextual_biasing.asr import make_asr
  asr = make_asr()                       # remote /chughtai by default
  asr = make_asr(backend="local")
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import wave
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REMOTE_BASE = "wss://ebitlogix-qwen-asr-vlm-async-test.hf.space"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# the /chughtai variant occasionally emits its own domain PROMPT on silence
# (VAD fires on non-speech). These signatures identify a leaked-prompt segment.
_LEAK = re.compile(
    r"diagnostic laib|kolars? spek|miks with inglish|paikistani diagnostic|"
    r"callers speak|urdoo/hindi|phone call .*chugtaai lab",
    re.IGNORECASE)


def _pcm16_24k(path) -> bytes:
    w = wave.open(str(path), "rb"); sr = w.getframerate(); n = w.getnframes()
    raw = w.readframes(n); w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if a.size and sr != 24000:
        x = np.linspace(0, 1, len(a), dtype=np.float64)
        xn = np.linspace(0, 1, int(round(len(a) * 24000 / sr)), dtype=np.float64)
        a = np.interp(xn, x, a)
    return a.astype(np.int16).tobytes()


# ── remote (WebSocket, OpenAI-Realtime) ──────────────────────────────────────
class RemoteASR:
    def __init__(self, variant: str = "en", base: str = REMOTE_BASE,
                 first_timeout: float = 40.0, quiet_timeout: float = 6.0,
                 max_wait: float = 120.0, retries: int = 1, verbose: bool = False):
        self.url = f"{base}/{variant}/v1/realtime"
        self.first_timeout = first_timeout        # wait this long for the FIRST segment
        self.quiet_timeout = quiet_timeout        # then this long between segments
        self.max_wait = max_wait
        self.retries = retries
        self.verbose = verbose

    def transcribe(self, audio, context: str = "", language: str | None = "Hindi") -> str:
        for attempt in range(self.retries + 1):
            try:
                out = asyncio.run(self._run(audio, context, language))
                if out or attempt == self.retries:
                    return out                    # retry once if empty (transient server queue)
            except Exception as e:
                if attempt == self.retries:
                    raise
                if self.verbose:
                    print(f"   retry after {type(e).__name__}", flush=True)
        return ""

    async def _run(self, audio, context: str, language: str | None = "Hindi") -> str:
        import websockets
        pcm = _pcm16_24k(audio)
        segments: list[str] = []
        async with websockets.connect(self.url, open_timeout=30, max_size=None,
                                      ping_interval=20) as ws:
            await ws.recv()                                  # session.created
            upd = {"type": "session.update", "session": {"input_audio_format": "pcm16"}}
            instructions_parts = []
            if language:
                instructions_parts.append(f"Transcribe in {language} script.")
            if context:
                instructions_parts.append(f"Domain terms likely spoken: {context}")
            if instructions_parts:
                upd["session"]["instructions"] = " ".join(instructions_parts)
            await ws.send(json.dumps(upd))
            step = 24000 * 2 // 2                             # ~0.5s frames
            for i in range(0, len(pcm), step):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i + step]).decode()}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))
            t0 = last = time.time()
            got_first = False
            while time.time() - t0 < self.max_wait:
                timeout = self.quiet_timeout if got_first else self.first_timeout
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break                                    # quiet -> done
                d = json.loads(m)
                if d.get("type") == "conversation.item.input_audio_transcription.completed":
                    txt = (d.get("transcript") or d.get("text") or "").strip()
                    if txt and not _LEAK.search(txt):
                        segments.append(txt)
                        if self.verbose:
                            print("   seg:", txt[:80], flush=True)
                    got_first = True
                    last = time.time()
                if got_first and time.time() - last > self.quiet_timeout:
                    break
        return " ".join(segments)


# ── local (qwen_asr, CPU) ────────────────────────────────────────────────────
class LocalASR:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu",
                 max_new_tokens: int = 256, verbose: bool = True):
        self.model_id, self.device = model_id, device
        self.max_new_tokens, self.verbose = max_new_tokens, verbose
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel
        if self.verbose:
            print(f"[asr] loading {self.model_id} on {self.device} ...", flush=True)
        t0 = time.time()
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id, dtype=torch.float32, device_map=self.device,
            max_new_tokens=self.max_new_tokens)
        if self.verbose:
            print(f"[asr] loaded in {time.time()-t0:.0f}s", flush=True)

    def transcribe(self, audio, context: str = "", language: str = "Hindi") -> str:
        self._ensure()
        out = self._model.transcribe(audio=[str(audio)], context=[context], language=[language])
        return out[0].text if out else ""


def make_asr(backend: str = "remote", **kw):
    """backend: 'remote' | 'local' | 'gpu-remote'."""
    if backend == "remote":
        return RemoteASR(**kw)
    if backend == "local":
        return LocalASR(**kw)
    if backend == "gpu-remote":
        from acoustic_contextual_biasing.gpu_remote_asr import GpuRemoteASR
        return GpuRemoteASR(**kw)
    raise ValueError(f"unknown backend {backend!r}")


# back-compat alias used by earlier scripts
BiasedASR = LocalASR
