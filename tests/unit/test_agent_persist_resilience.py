"""A dead/slow session store must not crash a live call — persistence is
best-effort (degrade to no-persistence). Regression guard for the deployment
incident where a Redis outage dropped Twilio S2S calls at agent.start()."""

from __future__ import annotations

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema


class _BoomStore:
    """A SessionStore whose every write raises (simulates a Redis outage)."""

    async def set_state(self, *a, **k):
        raise RuntimeError("redis down")

    async def append_history(self, *a, **k):
        raise RuntimeError("redis down")


def _agent(store):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1"),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="A", agent_role="s", company_name="X"),
        engine=object(), store=store)


@pytest.mark.asyncio
async def test_persist_state_swallows_store_failure():
    agent = _agent(_BoomStore())
    # Must not raise even though the store is dead.
    await agent.persist_state()
    await agent.persist_turn("user", "hello")


@pytest.mark.asyncio
async def test_agent_start_survives_dead_store():
    agent = _agent(_BoomStore())
    await agent.start()                      # calls persist_state internally
    # The call can still proceed: state advanced despite the dead store.
    from src.agents.state_machine import State
    assert agent.state.state is State.LISTENING
