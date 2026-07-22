"""
Qwen3-ASR GPU Remote — WebSocket server + client in one file.

SERVER (run on GPU machine):
  pip install websockets
  python acoustic_contextual_biasing/gpu_remote_asr.py --device cuda
  python acoustic_contextual_biasing/gpu_remote_asr.py --device mps

CLIENT (import from benchmark):
  from acoustic_contextual_biasing.gpu_remote_asr import GpuRemoteASR
  asr = GpuRemoteASR(url="ws://192.168.99.117:8910")
  text = asr.transcribe("audio.wav", context="Muhammad, Ahsan", language="Hindi")

Protocol (JSON over WebSocket):
  Client → {"audio": "<base64 wav>", "context": "Muhammad, Ahsan", "language": "Hindi"}
  Server → {"text": "...", "elapsed": 1.23}

Benchmark usage:
  python benchmark/benchmark_acoustic_biasing.py --transcribe --calls 20 \\
      --backend gpu-remote --gpu-url ws://192.168.99.117:8910
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time


MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


# ── CLIENT ───────────────────────────────────────────────────────────────────

class GpuRemoteASR:
    """WebSocket client for the GPU ASR server — supports context= biasing."""

    def __init__(self, url: str = "ws://192.168.99.117:8910"):
        self.url = url
        print(f"[gpu-remote] Will connect to {url}", flush=True)

    def transcribe(self, audio, context: str = "", language: str = "Hindi") -> str:
        return asyncio.run(self._send(audio, context, language))

    async def _send(self, audio, context: str, language: str) -> str:
        import websockets

        with open(str(audio), "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        async with websockets.connect(self.url, open_timeout=30,
                                      max_size=50 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "audio": audio_b64,
                "context": context,
                "language": language,
            }))
            resp = json.loads(await ws.recv())
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            return resp.get("text", "")


# ── SERVER ───────────────────────────────────────────────────────────────────

_model = None


def _load_model(device: str = "cuda", max_new_tokens: int = 256):
    global _model
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[asr_server] Loading {MODEL_ID} on {device} (dtype={dtype}) ...", flush=True)
    t0 = time.time()
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device, max_new_tokens=max_new_tokens)
    print(f"[asr_server] Loaded in {time.time()-t0:.0f}s", flush=True)


async def _handle_client(websocket):
    async for message in websocket:
        try:
            req = json.loads(message)
            audio_b64 = req.get("audio", "")
            context = req.get("context", "")
            language = req.get("language", "Hindi")

            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            t0 = time.time()
            try:
                out = _model.transcribe(
                    audio=[tmp_path], context=[context], language=[language])
                text = out[0].text if out else ""
            finally:
                os.unlink(tmp_path)

            elapsed = round(time.time() - t0, 3)
            await websocket.send(json.dumps({
                "text": text, "elapsed": elapsed
            }, ensure_ascii=False))

        except Exception as e:
            await websocket.send(json.dumps({
                "error": str(e), "text": ""
            }))


async def _run_server(host: str, port: int):
    import websockets
    print(f"[asr_server] WebSocket listening on ws://{host}:{port}", flush=True)
    async with websockets.serve(_handle_client, host, port,
                                max_size=50 * 1024 * 1024):
        await asyncio.Future()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Qwen3-ASR WebSocket GPU server")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8910)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    _load_model(device=args.device, max_new_tokens=args.max_new_tokens)
    asyncio.run(_run_server(args.host, args.port))


if __name__ == "__main__":
    main()
