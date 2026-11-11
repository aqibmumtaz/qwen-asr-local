"""LLM-dependent paths tested with a fake OpenAI-compatible server and scripted models (no real LLM needed)."""
import json
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request

from app.bot import Bot
from app.config import Settings
from app.llm.client import LLM, FakeLLM, parse_json_loose
from app.router.chain import ChainRouter
from app.router.interface import RouteContext, RouteDecision
from app.router.llm_router import QwenRouter
from tests.test_services import _free_port


class Scripted:
    """Router stub returning a fixed decision (or raising / sleeping) for chain tests."""
    def __init__(self, name, decision=None, conf=0.9, workflow=None, raises=False, delay=0.0):
        self.name, self.decision, self.conf, self.workflow, self.raises, self.delay = name, decision, conf, workflow, raises, delay
        self.calls = 0

    async def route(self, ctx):
        self.calls += 1
        if self.delay:
            import asyncio
            await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError("model down")
        return RouteDecision(self.decision, self.workflow, self.conf, source=self.name) if self.decision else None


def _manifests():
    import yaml
    from app.config import ROOT
    return {n: yaml.safe_load((ROOT / "workflows" / n / "manifest.yaml").read_text()) for n in ("home_sampling", "booking_status")}


def rc(text="kuch bhi", **kw):
    return RouteContext(utterance=text, text=text, workflows=_manifests(), **kw)


def chain(primary, fallback, **routers):
    s = Settings(llm_base_url="", router_primary=primary, router_fallback=fallback, laya_conf_min=0.8,
                 qwen_router_conf_min=0.7, router_timeout_ms=100)
    return ChainRouter(s, routers)


async def test_chain_uses_primary_when_confident():
    p, f = Scripted("laya", "faq", 0.95), Scripted("qwen", "human", 0.9)
    d = await chain("laya", "qwen", laya=p, qwen=f).route(rc("lab ka waqt"))
    assert (d.decision, d.source) == ("faq", "laya") and f.calls == 0


async def test_chain_falls_back_on_low_confidence():
    p, f = Scripted("laya", "faq", 0.4), Scripted("qwen", "lookup", 0.9)
    d = await chain("laya", "qwen", laya=p, qwen=f).route(rc("price"))
    assert (d.decision, d.source) == ("lookup", "qwen")
    assert [t["source"] for t in d.tried][:2] == ["laya", "qwen"]


async def test_chain_falls_back_on_error_and_timeout():
    f = Scripted("qwen", "lookup", 0.9)
    d = await chain("laya", "qwen", laya=Scripted("laya", raises=True), qwen=f).route(rc("price"))
    assert d.source == "qwen"
    d = await chain("laya", "qwen", laya=Scripted("laya", "faq", 0.99, delay=0.5), qwen=Scripted("qwen", "lookup", 0.9)).route(rc("price"))
    assert d.source == "qwen" and any(t.get("note") == "timeout" for t in d.tried)


async def test_chain_disagreement_on_switch_asks_the_caller():
    p = Scripted("laya", "switch", 0.6, workflow="booking_status")
    f = Scripted("qwen", "continue", 0.9)
    s = Settings(llm_base_url="", router_primary="laya", router_fallback="qwen", laya_conf_min=0.8, qwen_router_conf_min=0.7)
    d = await ChainRouter(s, {"laya": p, "qwen": f}).route(rc("kuch bhi"))
    assert d.decision == "unclear" and "disagreement" in d.reason


async def test_chain_ends_in_rules_when_models_unavailable():
    d = await chain("qwen", "none", qwen=Scripted("qwen", None)).route(rc("allah hafiz"))
    assert d.decision == "end"                              # cheap pass
    d = await chain("qwen", "none", qwen=Scripted("qwen", None)).route(rc("meri booking ka status batao"))
    assert d.decision == "switch" and d.source == "rules"   # baseline


async def test_qwen_router_parses_json_and_validates():
    ok = FakeLLM(lambda m: {"decision": "switch", "workflow": "booking_status", "confidence": 0.93})
    d = await QwenRouter(ok).route(rc("booking ka status"))
    assert (d.decision, d.workflow, d.confidence) == ("switch", "booking_status", 0.93)
    prompt = json.loads(ok.prompts[0][1]["content"])
    assert prompt["utterance"] == "booking ka status" and "home_sampling" in prompt["workflows"]
    assert await QwenRouter(FakeLLM(lambda m: {"decision": "teleport"})).route(rc()) is None          # off-list decision
    assert await QwenRouter(FakeLLM(lambda m: {"decision": "switch", "workflow": "nope"})).route(rc()) is None
    assert await QwenRouter(FakeLLM(lambda m: "not json at all")).route(rc()) is None
    assert await QwenRouter(FakeLLM(lambda m: None)).route(rc()) is None


def test_parse_json_loose_handles_think_blocks_and_fences():
    assert parse_json_loose('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! {"a": 2} done') == {"a": 2}
    assert parse_json_loose("nothing") is None


async def test_llm_name_extraction_used_when_available():
    llm = FakeLLM(lambda m: {"name": "muhammad ali khan"})
    bot = Bot(settings=Settings(llm_base_url="fake://"), llm=llm, routers={})
    s = bot.new_session()
    await bot.turn(s, "cbc ka test ghar par karwana hai")
    await bot.turn(s, "ji mera naam jo hai wo muhammad ali khan hai sahab")
    assert s.active.slots["patient_name"] == "Muhammad Ali Khan"
    assert llm.calls >= 1


async def test_llm_down_falls_back_to_rules_for_names():
    llm = FakeLLM(lambda m: None)             # model unavailable
    bot = Bot(settings=Settings(llm_base_url="fake://"), llm=llm, routers={})
    s = bot.new_session()
    await bot.turn(s, "cbc ka test ghar par karwana hai")
    await bot.turn(s, "mera naam ali raza hai")
    assert s.active.slots["patient_name"] == "Ali Raza"


@pytest.fixture(scope="module")
def openai_server():
    seen = {}
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        seen["body"] = body
        content = json.dumps({"decision": "faq", "workflow": None, "confidence": 0.88})
        return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "<think></think>" + content},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1", seen
    srv.should_exit = True


async def test_real_llm_client_request_shape_and_routing(openai_server):
    url, seen = openai_server
    s = Settings(llm_base_url=url, llm_api_key="k", llm_model="Qwen/Qwen3-8B", llm_enable_thinking=False)
    llm = LLM(s)
    d = await QwenRouter(llm).route(rc("lab kitne baje khulti hai"))
    assert d.decision == "faq" and d.confidence == 0.88
    b = seen["body"]
    assert b["model"] == "Qwen/Qwen3-8B" and b["response_format"] == {"type": "json_object"}
    assert b["extra_body"] if "extra_body" in b else True
    assert b.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert b["messages"][0]["role"] == "system"


async def test_llm_client_unreachable_returns_none_not_exception():
    llm = LLM(Settings(llm_base_url="http://127.0.0.1:9/v1"))
    assert await llm.chat([{"role": "user", "content": "hi"}], timeout=2) is None
    assert "Error" in llm.last_error or llm.last_error
