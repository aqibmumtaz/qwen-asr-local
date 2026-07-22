"""
Quick test client for the GPU remote ASR server.

Usage:
  # Just check the server is reachable (no transcription)
  python test_gpu_remote_asr.py

  # Transcribe a real audio file
  python test_gpu_remote_asr.py --audio audio.wav --context "Chughtai Lab, CBC, RFT" --language Hindi
"""
import argparse
import time

from gpu_remote_asr import GpuRemoteASR

def main():
    ap = argparse.ArgumentParser(description="Test client for gpu_remote_asr server")
    ap.add_argument("--url", default="ws://192.168.99.117:8910")
    ap.add_argument("--audio", default=None, help="Path to a .wav file to transcribe")
    ap.add_argument("--context", default="Chughtai Lab, CBC, RFT")
    ap.add_argument("--language", default="Hindi")
    args = ap.parse_args()

    asr = GpuRemoteASR(url=args.url)

    if args.audio is None:
        print("No --audio given, checking connectivity only...")
        import asyncio
        import websockets

        async def ping():
            async with websockets.connect(args.url, open_timeout=10) as ws:
                print(f"Connected OK to {args.url}")

        asyncio.run(ping())
        return

    t0 = time.time()
    text = asr.transcribe(args.audio, context=args.context, language=args.language)
    elapsed = time.time() - t0

    print(f"Transcript: {text}")
    print(f"Round-trip time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
