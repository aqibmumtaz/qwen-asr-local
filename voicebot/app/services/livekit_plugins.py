"""LiveKit Agents plugins that wrap our STT/TTS endpoints (LiveKit Agents 1.8.x API).

BotSTT  non-streaming recognizer (wrapped in StreamAdapter + Silero VAD, so each utterance is POSTed once)
BotTTS  chunked synthesizer that streams PCM from the OmniVoice websocket; cancelling playback closes the socket
"""
from __future__ import annotations

import uuid

from livekit.agents import APIConnectOptions, stt, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from app.config import Settings, get_settings
from app.services.stt import make_stt
from app.services.tts import WsTts


class BotSTT(stt.STT):
    def __init__(self, settings: Settings | None = None):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.s = settings or get_settings()
        self.client = make_stt(self.s)
        self.last_ms = 0
        self.last_error = ""

    @property
    def provider(self) -> str:
        return "chughtai-qwen3-asr"

    async def _recognize_impl(self, buffer: utils.AudioBuffer, *, language=None, conn_options: APIConnectOptions) -> stt.SpeechEvent:
        frame = utils.merge_frames(buffer)
        res = await self.client.transcribe_pcm(bytes(frame.data), frame.sample_rate)
        self.last_ms, self.last_error = res.ms, res.error
        alts = [stt.SpeechData(language="ur", text=res.text)] if res.text else [stt.SpeechData(language="ur", text="")]
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, request_id=uuid.uuid4().hex, alternatives=alts)


class BotTTS(tts.TTS):
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=s.tts_sample_rate, num_channels=1)
        self.s = s
        self.client = WsTts(s)

    @property
    def provider(self) -> str:
        return "omnivoice"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "BotChunkedStream":
        return BotChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class BotChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: BotTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._bt = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(request_id=uuid.uuid4().hex, sample_rate=self._bt.s.tts_sample_rate, num_channels=1,
                                  mime_type="audio/pcm")
        async for chunk in self._bt.client.stream(self._input_text):
            output_emitter.push(chunk)
        if self._bt.client.last.error and not self._bt.client.last.bytes:
            raise RuntimeError(f"tts failed: {self._bt.client.last.error}")
        output_emitter.flush()
