"""Router chain: cheap rules first, then primary (Laya) -> fallback (Qwen) -> rules baseline.

Fallback triggers (ROUTER_FALLBACK_ON): error, timeout, low_confidence.
If primary and fallback disagree on a switch, the chain returns `unclear` so the bot asks the caller.
"""
from __future__ import annotations

import asyncio
import time

from app.config import Settings
from app.router.interface import RouteContext, RouteDecision
from app.router.rules import RulesRouter, cheap_pass


class ChainRouter:
    def __init__(self, settings: Settings, routers: dict[str, object]):
        self.s = settings
        self.routers = routers
        self.baseline = RulesRouter()
        self.stats = {"calls": 0, "primary": 0, "fallback": 0, "baseline": 0, "cheap": 0}

    def _get(self, name: str):
        return self.routers.get(name) if name and name != "none" else None

    async def _try(self, router, ctx: RouteContext, min_conf: float, tried: list[dict], timeout_ms: int | None):
        t0 = time.perf_counter()
        trig = self.s.router_fallback_on
        try:
            coro = router.route(ctx)
            d = await asyncio.wait_for(coro, timeout=(timeout_ms / 1000.0) if timeout_ms else None)
        except asyncio.TimeoutError:
            tried.append({"source": router.name, "note": "timeout", "ms": int((time.perf_counter() - t0) * 1000)})
            return None, "timeout" in trig
        except Exception as e:  # model missing, runtime error
            tried.append({"source": router.name, "note": f"error: {type(e).__name__}: {str(e)[:80]}"})
            return None, "error" in trig
        ms = int((time.perf_counter() - t0) * 1000)
        if d is None:
            tried.append({"source": router.name, "note": "unavailable", "ms": ms})
            return None, "error" in trig
        tried.append({"source": router.name, "decision": d.decision, "workflow": d.workflow,
                      "confidence": round(d.confidence, 3), "ms": ms})
        if d.confidence < min_conf:
            return d, "low_confidence" in trig        # (decision, should_fall_back)
        return d, False

    async def route(self, ctx: RouteContext) -> RouteDecision:
        self.stats["calls"] += 1
        tried: list[dict] = []
        d = cheap_pass(ctx)
        if d:
            d.tried = [{"source": "rules", "decision": d.decision, "confidence": d.confidence, "note": d.reason}]
            self.stats["cheap"] += 1
            return d
        primary = self._get(self.s.router_primary)
        fallback = self._get(self.s.router_fallback)
        p_dec = None
        if primary is not None:
            min_p = self.s.laya_conf_min if primary.name == "laya" else self.s.qwen_router_conf_min
            p_dec, fall = await self._try(primary, ctx, min_p, tried, self.s.router_timeout_ms if primary.name == "laya" else None)
            if p_dec is not None and not fall:
                p_dec.tried = tried
                self.stats["primary"] += 1
                return p_dec
        if fallback is not None and fallback is not primary:
            f_dec, fall = await self._try(fallback, ctx, self.s.qwen_router_conf_min, tried, None)
            if f_dec is not None and not fall:
                if p_dec is not None and "switch" in (p_dec.decision, f_dec.decision) and p_dec.decision != f_dec.decision \
                        and p_dec.confidence >= 0.5:
                    out = RouteDecision("unclear", confidence=0.5, source="chain", reason="router disagreement on switch")
                    out.tried = tried
                    return out
                f_dec.tried = tried
                self.stats["fallback"] += 1
                return f_dec
        b = await self.baseline.route(ctx)
        tried.append({"source": "rules", "decision": b.decision, "workflow": b.workflow, "confidence": b.confidence,
                      "note": b.reason})
        b.tried = tried
        self.stats["baseline"] += 1
        return b
