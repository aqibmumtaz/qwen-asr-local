"""Settings loaded from .env (see .env.example). One place to read configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _s(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    return val.split("#", 1)[0].strip() if val is not None else default


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(float(_s(key, str(default))))
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    return _s(key, str(default)).lower() in ("1", "true", "yes", "on")


def _p(key: str, default: str) -> Path:
    return (ROOT / _s(key, default)).resolve()


def _list(key: str, default: str = "") -> list[str]:
    return [x.strip() for x in _s(key, default).split(",") if x.strip()]


@dataclass
class Settings:
    # endpoints
    stt_url: str = field(default_factory=lambda: _s("QWEN_VLLM_BASE_URL"))
    stt_token: str = field(default_factory=lambda: _s("QWEN_VLLM_TOKEN"))
    tts_url: str = field(default_factory=lambda: _s("OMNI_TTS"))
    tts_token: str = field(default_factory=lambda: _s("OMNI_TTS_TOKEN"))
    llm_base_url: str = field(default_factory=lambda: _s("LLM_BASE_URL"))
    llm_api_key: str = field(default_factory=lambda: _s("LLM_API_KEY") or "none")
    llm_model: str = field(default_factory=lambda: _s("LLM_MODEL", "Qwen/Qwen3-8B"))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 200))
    llm_enable_thinking: bool = field(default_factory=lambda: _b("LLM_ENABLE_THINKING", False))
    stt_mode: str = field(default_factory=lambda: _s("STT_MODE", "http"))
    stt_http_url: str = field(default_factory=lambda: _s("STT_HTTP_URL", "https://ebitlogix-qwen-asr-vlm-async-test.hf.space"))
    stt_variant: str = field(default_factory=lambda: _s("STT_VARIANT", "chughtai"))
    stt_timeout: float = field(default_factory=lambda: _f("STT_TIMEOUT_S", 30.0))
    tts_format: str = field(default_factory=lambda: _s("TTS_FORMAT", "auto"))
    tts_text_field: str = field(default_factory=lambda: _s("TTS_TEXT_FIELD", "text"))
    tts_timeout: float = field(default_factory=lambda: _f("TTS_TIMEOUT_S", 30.0))
    tts_sample_rate: int = field(default_factory=lambda: _i("TTS_SAMPLE_RATE", 24000))
    stt_sample_rate: int = field(default_factory=lambda: _i("STT_SAMPLE_RATE", 16000))
    # livekit
    livekit_url: str = field(default_factory=lambda: _s("LIVEKIT_URL", "ws://localhost:7880"))
    livekit_key: str = field(default_factory=lambda: _s("LIVEKIT_API_KEY", "devkey"))
    livekit_secret: str = field(default_factory=lambda: _s("LIVEKIT_API_SECRET", "secret"))
    # router
    router_primary: str = field(default_factory=lambda: _s("ROUTER_PRIMARY", "qwen"))
    router_fallback: str = field(default_factory=lambda: _s("ROUTER_FALLBACK", "none"))
    router_fallback_on: list[str] = field(
        default_factory=lambda: _list("ROUTER_FALLBACK_ON", "error,timeout,low_confidence"))
    router_timeout_ms: int = field(default_factory=lambda: _i("ROUTER_TIMEOUT_MS", 400))
    laya_model: str = field(default_factory=lambda: _s("LAYA_MODEL", "laya-multilingual"))
    laya_device: str = field(default_factory=lambda: _s("LAYA_DEVICE", "auto"))
    laya_conf_min: float = field(default_factory=lambda: _f("LAYA_CONFIDENCE_MIN", 0.80))
    laya_temperature: float = field(default_factory=lambda: _f("LAYA_TEMPERATURE", 1.0))
    qwen_router_conf_min: float = field(default_factory=lambda: _f("ROUTER_QWEN_CONFIDENCE_MIN", 0.70))
    switch_conf_min: float = field(default_factory=lambda: _f("ROUTER_SWITCH_CONFIDENCE_MIN", 0.85))
    router_log: bool = field(default_factory=lambda: _b("ROUTER_LOG_DECISIONS", True))
    # entities
    entity_config_dir: Path = field(default_factory=lambda: _p("ENTITY_CONFIG_DIR", "config/entities"))
    entity_data_dir: Path = field(default_factory=lambda: _p("ENTITY_DATA_DIR", "data"))
    ent_accept: float = field(default_factory=lambda: _f("ENTITY_AUTO_ACCEPT_SCORE", 0.90))
    ent_gap: float = field(default_factory=lambda: _f("ENTITY_ACCEPT_GAP", 0.15))
    ent_confirm: float = field(default_factory=lambda: _f("ENTITY_CONFIRM_SCORE", 0.70))
    ent_max_choices: int = field(default_factory=lambda: _i("ENTITY_MAX_CHOICES", 3))
    ent_max_rounds: int = field(default_factory=lambda: _i("ENTITY_MAX_ROUNDS", 2))
    # dialog
    max_slot_retries: int = field(default_factory=lambda: _i("MAX_SLOT_RETRIES", 2))
    max_suspended: int = field(default_factory=lambda: _i("MAX_SUSPENDED_WORKFLOWS", 2))
    human_number: str = field(default_factory=lambda: _s("HUMAN_TRANSFER_NUMBER"))
    faq_file: Path = field(default_factory=lambda: _p("FAQ_FILE", "config/faq.yaml"))
    tts_script: str = field(default_factory=lambda: _s("TTS_SCRIPT", "roman"))
    # backend / safety
    booking_api_mode: str = field(default_factory=lambda: _s("BOOKING_API_MODE", "mock"))
    shadow_mode: bool = field(default_factory=lambda: _b("SHADOW_MODE", False))
    redact_pii: bool = field(default_factory=lambda: _b("REDACT_PII_IN_LOGS", True))
    log_dir: Path = field(default_factory=lambda: _p("LOG_DIR", "logs"))
    tz: str = field(default_factory=lambda: _s("TZ", "Asia/Karachi"))
    ui_port: int = field(default_factory=lambda: _i("UI_PORT", 7860))
    vad_stop_secs: float = field(default_factory=lambda: _f("VAD_STOP_SECS", 0.5))
    allow_interruptions: bool = field(default_factory=lambda: _b("ALLOW_INTERRUPTIONS", True))
    agent_idle_procs: int = field(default_factory=lambda: _i("AGENT_IDLE_PROCESSES", 2))
    agent_init_timeout: float = field(default_factory=lambda: _f("AGENT_INIT_TIMEOUT_S", 60.0))
    # persona
    bot_name: str = field(default_factory=lambda: _s("BOT_NAME", "Zara"))
    bot_female: bool = field(default_factory=lambda: _b("BOT_FEMALE", True))


def get_settings() -> Settings:
    return Settings()
