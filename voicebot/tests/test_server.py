"""API tests for the console backend (chat flow, guided scenarios, scenario runner, token minting)."""
import pytest
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_index_and_static():
    assert client.get("/").status_code == 200
    assert "Voice bot console" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_config_reports_router_and_workflows():
    c = client.get("/api/config").json()
    assert set(c["workflows"]) == {"home_sampling", "booking_status"}
    assert c["router"]["primary"] in ("laya", "qwen", "none")


def test_chat_session_flow():
    s = client.post("/api/session", json={"greet": True}).json()
    assert "home sampling" in s["greeting"]
    r = client.post("/api/turn", json={"session_id": s["session_id"], "text": "cbc ka test ghar par karwana hai"}).json()
    assert r["route"]["decision"] == "switch" and r["state"]["active"]["step"] == "patient_name"
    assert r["timings"]["total_ms"] >= 0 and any(e["kind"] == "route" for e in r["events"])
    r = client.post("/api/turn", json={"session_id": s["session_id"], "text": "ali raza"}).json()
    assert r["state"]["active"]["slots"]["patient_name"] == "Ali Raza"


def test_unknown_session_and_ended_session():
    assert client.post("/api/turn", json={"session_id": "nope", "text": "x"}).status_code == 404
    s = client.post("/api/session", json={}).json()
    client.post("/api/turn", json={"session_id": s["session_id"], "text": "allah hafiz"})
    assert client.post("/api/turn", json={"session_id": s["session_id"], "text": "x"}).status_code == 409


def test_guided_scenario_reports_checks():
    sc = client.get("/api/scenarios").json()
    one = next(x for x in sc if x["id"] == "hs_one_sentence")
    s = client.post("/api/session", json={"scenario_id": "hs_one_sentence"}).json()
    r = client.post("/api/turn", json={"session_id": s["session_id"], "text": one["lines"][0]}).json()
    assert r["check"]["scripted"] and r["check"]["ok"]
    assert r["scenario_done"]["ok"]


def test_guided_line_not_scripted_is_not_checked():
    s = client.post("/api/session", json={"scenario_id": "hs_happy_path"}).json()
    r = client.post("/api/turn", json={"session_id": s["session_id"], "text": "kuch aur"}).json()
    assert r["check"]["scripted"] is False


def test_run_all_scenarios_via_api():
    res = client.post("/api/scenarios/run", json={"tag": "smoke"}).json()
    assert res["total"] >= 5 and res["passed"] == res["total"]


def test_backend_switch_fails_next_call():
    s = client.post("/api/session", json={}).json()
    sid = s["session_id"]
    for line in ["cbc ka test ghar par karwana hai", "ali raza", "painteesh saal", "mard", "dha karachi house 12",
                 "kal subah", "0300 1234567", "haan", "haan"][:-1]:
        client.post("/api/turn", json={"session_id": sid, "text": line})
    assert client.post("/api/backend", json={"session_id": sid, "fail_next": "down"}).json()["fail_next"] == "down"
    r = client.post("/api/turn", json={"session_id": sid, "text": "haan"}).json()
    assert r["transfer"] == "backend_error"


def test_token_endpoint():
    pytest.importorskip("livekit.api")
    t = client.post("/api/token").json()
    assert t["token"].count(".") == 2 and t["room"].startswith("call-")
