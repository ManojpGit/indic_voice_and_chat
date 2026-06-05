"""Outcome recording fires on the PRODUCTION bridge subclasses.

The bootstrap subclasses (_AgentBridge / _ExotelAgentBridge) override run()
with their own finally block, so this guards against the override silently
dropping the _record_outcome() call that the base classes have.
"""
from __future__ import annotations

import json

import pytest

from src.api.telephony_exotel import ExotelBridgeConfig
from src.api.telephony_twilio import TwilioBridgeConfig
from src.bootstrap import _AgentBridge, _ExotelAgentBridge
from src.campaign.models import CallAnalysis, LeadCallOutcome
from src.pipeline.vad import EnergyVAD


class _WS:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list[str] = []

    async def receive_text(self) -> str:
        if self._frames:
            return json.dumps(self._frames.pop(0))
        raise RuntimeError("ws closed")  # no more frames; only reached if not stopped

    async def send_text(self, s: str) -> None:
        self.sent.append(s)


class _Agent:
    def __init__(self) -> None:
        self.started = False
        self.opening_played = False
        self.hung_up = False

    async def start(self) -> None:
        self.started = True

    async def play_opening(self, sink) -> None:
        self.opening_played = True

    async def handle_turn(self, captured, sink):  # pragma: no cover - no media here
        return None

    async def handle_hangup(self) -> None:
        self.hung_up = True


def _vad() -> EnergyVAD:
    return EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)


@pytest.mark.asyncio
async def test_twilio_agentbridge_records_outcome_on_stop(monkeypatch) -> None:
    import src.api.outcome_recorder as orec

    calls = []

    async def fake_analyze(agent, **kw):
        calls.append(kw)
        return CallAnalysis(outcome=LeadCallOutcome.INTERESTED, summary="s")

    monkeypatch.setattr(orec, "analyze_agent_call", fake_analyze)

    agent = _Agent()
    ws = _WS([{"event": "connected"}, {"event": "start", "start": {"streamSid": "MZ"}}, {"event": "stop"}])
    bridge = _AgentBridge(
        websocket=ws, agent=agent, vad=_vad(), config=TwilioBridgeConfig(),
        llm=object(), tenant_timezone="Asia/Kolkata",
    )
    await bridge.run()
    assert agent.opening_played is True
    assert len(calls) == 1  # outcome recorded via the subclass finally
    assert agent.hung_up is True


@pytest.mark.asyncio
async def test_exotel_agentbridge_records_outcome_on_stop(monkeypatch) -> None:
    import src.api.outcome_recorder as orec

    calls = []

    async def fake_analyze(agent, **kw):
        calls.append(kw)
        return CallAnalysis(outcome=LeadCallOutcome.NOT_INTERESTED, summary="s")

    monkeypatch.setattr(orec, "analyze_agent_call", fake_analyze)

    agent = _Agent()
    ws = _WS([{"event": "connected"}, {"event": "start", "stream_sid": "EX"}, {"event": "stop"}])
    bridge = _ExotelAgentBridge(
        websocket=ws, agent=agent, vad=_vad(), config=ExotelBridgeConfig(),
        llm=object(), tenant_timezone="Asia/Kolkata",
    )
    await bridge.run()
    assert agent.opening_played is True
    assert len(calls) == 1
    assert agent.hung_up is True
