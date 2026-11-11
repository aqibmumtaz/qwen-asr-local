#!/usr/bin/env python3
"""End-to-end voice loop with FAKE STT/TTS (no tokens needed): LiveKit server + agent worker + simulated caller.

  caller (real call audio) -> LiveKit room -> agent: Silero VAD -> BotSTT -> fake STT HTTP (returns scripted lines)
     -> Bot.turn (router, workflow) -> BotTTS -> fake TTS websocket (sine tone) -> caller

Needs a running LiveKit server (docker run -d -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev).
Checks that each spoken line reaches the bot, the bot's reply comes back as audio, and the flow state advances.
Usage: python scripts/e2e_voice_fake.py [--lines N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402
from livekit import api, rtc  # noqa: E402

from app.config import ROOT  # noqa: E402

LINES = ["cbc ka test ghar par karwana hai", "ali raza", "painteesh saal", "mard", "dha karachi house 12", "kal subah"]
LK_URL = os.getenv("E2E_LIVEKIT_URL", "ws://localhost:7880")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def speech_clip(seconds: float = 3.0, rate: int = 16000) -> np.ndarray:
    """A slice of real call audio (loud part), resampled to 16 kHz int16."""
    hits = sorted((ROOT.parent / "qwen3-asr-local/benchmark/lab_test_80_audios_chunks_dynamic").glob("*/chunk_000.wav"))
    if not hits:
        raise SystemExit("no sample audio found under qwen3-asr-local/benchmark/lab_test_80_audios_chunks_dynamic")
    w = wave.open(str(hits[0]), "rb")
    sr, a = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    win = int(sr * seconds)
    step = sr // 2
    best = max(range(0, max(len(a) - win, 1), step), key=lambda i: float(np.abs(a[i:i + win]).mean()))
    seg = a[best:best + win]
    seg = np.interp(np.linspace(0, len(seg), int(len(seg) * rate / sr), endpoint=False), np.arange(len(seg)), seg)
    seg = seg / max(np.abs(seg).max(), 1) * 12000
    return seg.astype(np.int16)


class Fakes:
    def __init__(self, lines, slow=False):
        self.slow = slow
        self.lines, self.stt_calls, self.tts_texts, self.tts_cancelled = list(lines), 0, [], 0
        self.stt_port, self.tts_port = free_port(), free_port()

    def start(self):
        app = FastAPI()

        @app.post("/chughtai/v1/audio/transcriptions")
        async def tr(file: UploadFile = File(...)):
            await file.read()
            i = self.stt_calls
            self.stt_calls += 1
            txt = self.lines[i] if i < len(self.lines) else ""
            return {"text": txt, "romanized_text": txt}

        self.srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.stt_port, log_level="warning"))
        threading.Thread(target=self.srv.run, daemon=True).start()
        while not self.srv.started:
            time.sleep(0.1)

        async def tts_handler(ws):
            req = json.loads(await ws.recv())
            self.tts_texts.append(req.get("text", ""))
            n = int(24000 * (min(2.0 + 0.1 * len(req.get("text", "")), 9.0) if self.slow else min(0.4 + 0.03 * len(req.get("text", "")), 3.0)))
            t = np.arange(n) / 24000.0
            pcm = (4000 * np.sin(2 * math.pi * 330 * t)).astype(np.int16).tobytes()
            try:
                for i in range(0, len(pcm), 9600):
                    await ws.send(pcm[i:i + 9600])
                    await asyncio.sleep(0.35 if self.slow else 0.02)   # real-time pacing in barge-in mode
                await ws.close()
            except websockets.ConnectionClosed:
                self.tts_cancelled += 1

        def run_ws():
            async def main():
                async with websockets.serve(tts_handler, "127.0.0.1", self.tts_port):
                    await asyncio.Future()
            asyncio.run(main())
        threading.Thread(target=run_ws, daemon=True).start()
        time.sleep(0.5)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=len(LINES))
    ap.add_argument("--scenario", help="replay this scenario id from eval/scenarios through audio")
    ap.add_argument("--barge-in", action="store_true", help="caller talks over the bot reply; the reply must be cut off")
    a = ap.parse_args()
    lines, expect_last = LINES[: a.lines], []
    if a.scenario:
        from app.scenarios import load_scenarios
        sc = next(x for x in load_scenarios() if x["id"] == a.scenario)
        lines = [st["say"] for st in sc["steps"]]
        expect_last = (sc["steps"][-1].get("expect") or {}).get("bot_contains", [])
    fakes = Fakes(lines, slow=a.barge_in)
    fakes.start()
    env = dict(os.environ, LIVEKIT_URL=LK_URL, LIVEKIT_API_KEY="devkey", LIVEKIT_API_SECRET="secret",
               STT_MODE="http", STT_HTTP_URL=f"http://127.0.0.1:{fakes.stt_port}", STT_VARIANT="chughtai",
               OMNI_TTS=f"ws://127.0.0.1:{fakes.tts_port}", LLM_BASE_URL="", ROUTER_PRIMARY="none", ROUTER_FALLBACK="none",
               LOG_DIR=str(ROOT / "logs" / "e2e"), PYTHONWARNINGS="ignore", AGENT_IDLE_PROCESSES="1")
    agent = subprocess.Popen([sys.executable, "-m", "app.agent", "start"], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    agent_log: list[str] = []
    threading.Thread(target=lambda: [agent_log.append(l) for l in agent.stdout], daemon=True).start()
    for _ in range(120):    # wait until the worker has registered with the LiveKit server
        if any("registered worker" in l for l in agent_log):
            break
        time.sleep(0.5)
    ok = True
    try:
        room_name = f"e2e-{int(time.time())}"
        tok = api.AccessToken("devkey", "secret").with_identity("caller").with_grants(
            api.VideoGrants(room_join=True, room=room_name)).to_jwt()
        room = rtc.Room()
        turns: list[dict] = []
        got_audio = {"frames": 0}
        done = asyncio.Event()

        @room.on("data_received")
        def on_data(pkt: rtc.DataPacket):
            if pkt.topic == "bot-turn":
                turns.append(json.loads(bytes(pkt.data)))

        @room.on("track_subscribed")
        def on_track(track, pub, part):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                async def drain():
                    async for ev in rtc.AudioStream(track):
                        got_audio["frames"] += 1
                asyncio.ensure_future(drain())

        await room.connect(LK_URL, tok)
        src = rtc.AudioSource(16000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", src)
        await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        clip = speech_clip()

        async def speak(samples: np.ndarray):
            for i in range(0, len(samples), 160):
                chunk = samples[i:i + 160]
                if len(chunk) < 160:
                    chunk = np.pad(chunk, (0, 160 - len(chunk)))
                fr = rtc.AudioFrame.create(16000, 1, 160)
                np.frombuffer(fr.data, dtype=np.int16)[:] = chunk
                await src.capture_frame(fr)

        async def silence(sec: float):
            await speak(np.zeros(int(16000 * sec), dtype=np.int16))

        print("waiting for the agent to join and greet ...")
        for _ in range(60):
            if got_audio["frames"] > 5:
                break
            await silence(0.5)
        print(f"  greeting audio frames received: {got_audio['frames']}; fake TTS saw: {fakes.tts_texts[:1]}")
        ok &= got_audio["frames"] > 5
        if a.barge_in:
            await speak(clip)
            await silence(1.6)
            for _ in range(40):
                if turns:
                    break
                await silence(0.5)
            print("  turn 1 reply started:", turns[0]["text"][:60] if turns else None)
            await silence(0.4)                      # bot is now speaking; talk over it
            await speak(clip)
            await silence(1.6)
            for _ in range(40):
                if len(turns) >= 2:
                    break
                await silence(0.5)
            await silence(1.0)
            print(f"  turns: {len(turns)} | TTS streams cut off by the caller: {fakes.tts_cancelled} | STT calls: {fakes.stt_calls}")
            ok = ok and len(turns) >= 2 and fakes.tts_cancelled >= 1 and fakes.stt_calls >= 2
            await room.disconnect()
            lines = []
        for i, line in enumerate(lines):
            before = len(turns)
            await speak(clip)
            await silence(1.6)
            for _ in range(40):
                if len(turns) > before:
                    break
                await silence(0.5)
            if len(turns) > before:
                t = turns[-1]
                st = t["state"]["active"]
                print(f"  line {i + 1}: STT said {line!r} -> bot: {t['text'][:70]!r} | step={st['step'] if st else None}")
            else:
                print(f"  line {i + 1}: NO bot turn received")
                ok = False
            await silence(2.0)      # let the bot's reply finish playing
        ok &= (len(turns) == len(lines)) if lines else ok
        if a.scenario:
            for sub in expect_last:
                good = bool(turns) and sub.lower() in turns[-1]["text"].lower()
                print(f"  last reply contains {sub!r}: {good}")
                ok &= good
        elif turns:
            final = turns[-1]["state"]["active"]
            ok &= bool(final) and final["workflow"] == "home_sampling"
        print(f"\nSTT calls: {fakes.stt_calls} | TTS requests: {len(fakes.tts_texts)} | audio frames back: {got_audio['frames']}")
        await room.disconnect()
    finally:
        agent.terminate()
        try:
            agent.wait(timeout=10)
        except Exception:
            agent.kill()
        if not ok:
            print("\n--- agent log (tail) ---")
            print("".join(agent_log[-400:]))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
