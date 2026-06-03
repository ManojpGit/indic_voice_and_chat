from __future__ import annotations

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
