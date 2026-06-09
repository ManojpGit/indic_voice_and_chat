import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.telephony_hooks as hooks
from src.api.telephony_stringee_bridge import StringeeIvrBridge, pcm16_to_wav


class _Agent:
    def __init__(self):
        self.state = type("S", (), {"is_terminal": False})()
        self._action = "continue"
    async def start(self): pass
    async def play_opening(self, sink): await sink(b"\x01\x00" * 8)
    async def handle_turn(self, captured, sink):
        await sink(b"\x02\x00" * 8)
        return type("O", (), {"response": type("R", (), {"response_text": "ok", "action": self._action})()})()
    async def handle_hangup(self): pass


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(hooks.router, prefix="/api/v1")

    async def _fetch(url): return pcm16_to_wav(b"\x00\x00" * 80, 8000)

    def _factory(*, call_id, tenant, base_url, fetch):
        return StringeeIvrBridge(
            call_id=str(call_id), agent=_Agent(), llm=None,
            tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
            base_url=base_url, tenant_slug="dev", fetch=_fetch,
        )
    hooks.set_stringee_bridge_factory(_factory)

    from types import SimpleNamespace
    async def _resolve(num): return SimpleNamespace(slug="dev", id="dev")
    monkeypatch.setattr(hooks, "tenant_from_twilio_to_number", _resolve)
    yield TestClient(app)
    hooks.set_stringee_bridge_factory(None)


def test_answer_returns_opening_scco_and_audio_served(client):
    r = client.post("/api/v1/telephony/stringee/answer",
                    json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"
    token = scco[0]["url"].rsplit("/", 1)[1]
    a = client.get(f"/api/v1/telephony/stringee/audio/{token}")
    assert a.status_code == 200 and a.content[:4] == b"RIFF"


def test_event_runs_a_turn_and_returns_reply_scco(client):
    client.post("/api/v1/telephony/stringee/answer",
                json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=c1",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"


def test_event_unknown_call_returns_reprompt(client):
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=nope",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    assert r.json()[0]["action"] == "talk"
