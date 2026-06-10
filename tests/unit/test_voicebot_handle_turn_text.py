from __future__ import annotations

import asyncio

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.pipeline.engine import TurnMetrics, TurnResult


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
        self.calls.append(user_text)
        return self._result


def _agent(engine):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=engine,
        store=None,
    )


@pytest.mark.asyncio
async def test_history_window_bounds_llm_context():
    """The LLM gets system prompt + last MAX_HISTORY_TURNS exchanges, not the
    whole transcript — so per-turn latency doesn't grow with call length. The
    full transcript still lives in session.turns."""
    from src.agents.voicebot import MAX_HISTORY_TURNS
    from src.interfaces.llm import LLMMessage

    captured = {}

    class _CaptureEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            captured["history"] = list(history)
            return TurnResult(
                user_text=user_text, user_language="hi", user_confidence=1.0,
                agent_text='{"response_text": "ok", "action": "continue"}',
                audio_bytes_sent=1, metrics=TurnMetrics(),
            )

    agent = _agent(_CaptureEngine())
    await agent.start()
    # Simulate a long call: system prompt (added in __init__) + 20 prior messages.
    for i in range(10):
        agent.session.turns.append(LLMMessage(role="user", content=f"u{i}"))
        agent.session.turns.append(LLMMessage(role="assistant", content=f"a{i}"))

    async def sink(a):
        pass

    await agent.handle_turn_text("latest", sink)

    hist = captured["history"]
    # system prompt + 2*MAX_HISTORY_TURNS recent messages.
    assert hist[0].role == "system"
    assert len(hist) == 1 + 2 * MAX_HISTORY_TURNS
    # Oldest recent message is u4 (turns 0-3 dropped), newest is a9.
    assert hist[1].content == f"u{10 - MAX_HISTORY_TURNS}"
    assert hist[-1].content == "a9"
    # Full transcript is preserved (system + 20 prior + new user + new assistant).
    assert len(agent.session.turns) == 1 + 20 + 2


@pytest.mark.asyncio
async def test_handle_turn_text_records_and_advances():
    result = TurnResult(
        user_text="और कुछ benefits हैं?",
        user_language="hi",
        user_confidence=1.0,
        agent_text='{"response_text": "जी हाँ!", "action": "continue", "updated_slots": {}}',
        audio_bytes_sent=10,
        metrics=TurnMetrics(),
    )
    engine = _FakeEngine(result)
    agent = _agent(engine)
    await agent.start()

    sink_calls = []

    async def sink(a):
        sink_calls.append(a)

    outcome = await agent.handle_turn_text("और कुछ benefits हैं?", sink)
    assert engine.calls == ["और कुछ benefits हैं?"]
    assert outcome.response.response_text == "जी हाँ!"
    roles = [t.role for t in agent.session.turns]
    assert roles[-2:] == ["user", "assistant"]
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_empty_is_noop():
    result = TurnResult(
        user_text="", user_language=None, user_confidence=0.0,
        agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(),
    )
    agent = _agent(_FakeEngine(result))
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("", sink)
    assert outcome.response.parse_error == "empty STT"
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_recovers_on_provider_hang(monkeypatch):
    """A hung provider call must not wedge the agent: the per-turn timeout
    walks the state machine back to LISTENING with a timeout error."""
    import src.agents.voicebot as vb
    monkeypatch.setattr(vb, "TURN_TIMEOUT_S", 0.05)

    class _HangingEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            await asyncio.sleep(5)  # never returns within the timeout
            raise AssertionError("should have timed out")

    agent = _agent(_HangingEngine())
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("कुछ", sink)
    assert agent.state.state is State.LISTENING
    assert "TimeoutError" in (outcome.response.parse_error or "")


@pytest.mark.asyncio
async def test_handle_turn_text_barge_in_drops_agent_reply():
    """A cancelled (barge-in) turn keeps the user turn but drops the agent
    reply and returns to LISTENING."""
    result = TurnResult(
        user_text="और कुछ benefits हैं?",
        user_language="hi",
        user_confidence=1.0,
        agent_text='{"response_text": "जी हाँ, सुन',  # partial / abandoned
        audio_bytes_sent=4,
        metrics=TurnMetrics(),
        cancelled=True,
    )
    engine = _FakeEngine(result)
    agent = _agent(engine)
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("और कुछ benefits हैं?", sink)
    assert outcome.response.parse_error == "barge-in"
    assert outcome.response.response_text == ""
    roles = [t.role for t in agent.session.turns]
    assert roles[-1] == "user"          # user turn kept, no trailing assistant turn
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_passes_cancel_event_to_engine():
    captured = {}

    class _CaptureEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            captured["cancel_event"] = cancel_event
            return TurnResult("u", "hi", 1.0,
                              '{"response_text": "ok", "action": "continue"}', 0, TurnMetrics())

    agent = _agent(_CaptureEngine())
    await agent.start()
    sentinel = object()

    async def sink(a):
        pass

    await agent.handle_turn_text("hi", sink, cancel_event=sentinel)
    assert captured["cancel_event"] is sentinel
