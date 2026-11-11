"""LiveKit voice worker: caller audio -> VAD -> STT -> Bot.turn -> TTS -> caller audio.

Run (dev):   livekit-server --dev        # separate terminal
             python -m app.agent dev     # registers a worker; join a room from the console (Voice mode)
The bot logic is the same `Bot` used by the chat console; this file only adapts audio in/out.
"""
from __future__ import annotations

import asyncio
import json
import logging

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm
from livekit.plugins import silero

from app.bot import Bot
from app.config import get_settings
from app.services.livekit_plugins import BotSTT, BotTTS

log = logging.getLogger("voicebot.agent")
settings = get_settings()
bot = Bot(settings=settings)


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "type", "") == "message" and getattr(item, "role", "") == "user":
            return (item.text_content or "").strip()
    return ""


class NullLLM(llm.LLM):
    """Placeholder so AgentSession runs its reply pipeline; BotAgent.llm_node never calls it."""

    def chat(self, *, chat_ctx, tools=None, conn_options=None, parallel_tool_calls=None, tool_choice=None, extra_kwargs=None):
        raise RuntimeError("NullLLM must not be called: BotAgent.llm_node handles every turn")


class BotAgent(Agent):
    """Replaces the LLM node with the bot's turn loop (router, workflows, FAQ, lookups)."""

    def __init__(self, ctx: JobContext, bot_session):
        super().__init__(instructions="Roman Urdu lab assistant (logic is in Bot.turn).")
        self.ctx = ctx
        self.bs = bot_session

    async def llm_node(self, chat_ctx, tools, model_settings):
        text = _last_user_text(chat_ctx)
        if not text:
            return
        res = await bot.turn(self.bs, text)
        await self._publish(res, text)
        for r in res.replies:
            yield r + " "
        if res.ended:
            asyncio.create_task(self._finish(res.transfer))

    async def _publish(self, res, user_text: str = "") -> None:
        try:
            payload = {"user": user_text, "replies": res.replies, "text": res.text, "route": res.route, "events": res.events,
                       "state": res.state, "timings": res.timings, "ended": res.ended, "transfer": res.transfer}
            await self.ctx.room.local_participant.publish_data(json.dumps(payload, default=str).encode(), topic="bot-turn")
        except Exception as e:  # UI event stream is best-effort
            log.warning("could not publish turn event: %s", e)

    async def _finish(self, transfer: str | None) -> None:
        try:
            await asyncio.sleep(0.5)
            handle = self.session.current_speech
            if handle is not None:
                await asyncio.wait_for(handle.wait_for_playout(), timeout=20)
        except Exception:
            pass
        if transfer:
            log.info("transfer requested (%s): hand the call to a person here (SIP transfer in Phase 4)", transfer)
        self.session.shutdown()


def prewarm(proc: agents.JobProcess) -> None:
    """Runs once per worker process (not per call): load the VAD model."""
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=settings.vad_stop_secs, sample_rate=16000)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    caller = participant.attributes.get("sip.phoneNumber") if participant.attributes else None
    bs = bot.new_session(caller_phone=caller)
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=settings.vad_stop_secs, sample_rate=16000)
    session = AgentSession(
        stt=agents.stt.StreamAdapter(stt=BotSTT(settings), vad=vad),
        vad=vad,
        llm=NullLLM(),
        tts=BotTTS(settings),
        turn_detection="vad",
        allow_interruptions=settings.allow_interruptions,
        min_endpointing_delay=settings.vad_stop_secs,
    )
    await session.start(room=ctx.room, agent=BotAgent(ctx, bs))
    greeting = bot.greeting(bs)
    await session.say(greeting.text, allow_interruptions=True)


def main() -> None:
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                              num_idle_processes=settings.agent_idle_procs,
                              initialize_process_timeout=settings.agent_init_timeout))


if __name__ == "__main__":
    main()
