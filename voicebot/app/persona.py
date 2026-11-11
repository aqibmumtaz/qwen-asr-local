"""Gender-dependent verb forms for the bot voice (Urdu verbs agree with the speaker's gender)."""
from __future__ import annotations

from app.config import Settings


def persona_vars(s: Settings) -> dict[str, str]:
    if s.bot_female:
        return {"sakti": "sakti", "leti": "leti", "karti": "karti", "rahi": "rahi", "bot_name": s.bot_name,
                "pa": "pa rahi", "deti": "deti"}
    return {"sakti": "sakta", "leti": "leta", "karti": "karta", "rahi": "raha", "bot_name": s.bot_name,
            "pa": "pa raha", "deti": "deta"}
