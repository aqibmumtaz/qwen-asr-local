#!/usr/bin/env python3
"""Phase 0 endpoint check: reachability, auth, latency and message formats for STT, TTS and LLM.

Usage: python scripts/ping_endpoints.py [--wav path/to/16k_or_any.wav]
Never prints tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, get_settings  # noqa: E402
from app.llm.client import LLM  # noqa: E402
from app.services.stt import HttpStt, WsStt  # noqa: E402
from app.services.tts import WsTts  # noqa: E402

SAMPLE_GLOB = "qwen3-asr-local/benchmark/lab_test_80_audios_chunks_dynamic/*/chunk_000.wav"


def find_sample() -> Path | None:
    hits = sorted(ROOT.parent.glob(SAMPLE_GLOB))
    return hits[0] if hits else None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    a = ap.parse_args()
    s = get_settings()
    wav = Path(a.wav) if a.wav else find_sample()
    print(f"tokens: STT {'set' if s.stt_token else 'EMPTY'} | TTS {'set' if s.tts_token else 'EMPTY'} | "
          f"LLM url {'set' if s.llm_base_url else 'EMPTY'}\n")

    print("== STT (HTTP batch)", HttpStt(s).url)
    if wav and wav.exists():
        w = wave.open(str(wav), "rb")
        print("   sample:", wav.name, f"{w.getframerate()} Hz, {w.getnframes() / w.getframerate():.1f}s")
        pcm, rate = w.readframes(w.getnframes()), w.getframerate()
        r = await HttpStt(s).transcribe_pcm(pcm, rate)
        print(f"   -> {r.ms} ms", "ERROR " + r.error if r.error else f"text: {r.text[:100]!r}")
        print("== STT (websocket)", s.stt_url)
        r = await WsStt(s).transcribe_pcm(pcm, rate)
        print(f"   -> {r.ms} ms", "ERROR " + r.error if r.error else f"text: {r.text[:100]!r}")
    else:
        print("   no sample wav found (pass --wav)")

    print("\n== TTS", s.tts_url)
    t = WsTts(s)
    audio = await t.synthesize("Assalam o alaikum, Chughtai Lab mein khush aamdeed.")
    st = t.last
    print(f"   -> first audio {st.first_audio_ms} ms, total {st.total_ms} ms, {len(audio)} bytes",
          "ERROR " + st.error if st.error else "")
    for m in st.messages or []:
        print("   msg:", m)

    print("\n== LLM", s.llm_base_url or "(not configured)")
    llm = LLM(s)
    if llm.enabled:
        t0 = time.perf_counter()
        out = await llm.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=10)
        print(f"   -> {int((time.perf_counter() - t0) * 1000)} ms", repr(out) if out else "ERROR " + llm.last_error)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
