from __future__ import annotations

import pytest

from src.agents.state_machine import (
    AgentStateMachine,
    Event,
    InvalidTransition,
    State,
)


@pytest.mark.asyncio
async def test_full_happy_path() -> None:
    sm = AgentStateMachine()
    assert sm.state is State.IDLE
    await sm.fire(Event.CALL_CONNECTED)
    assert sm.state is State.LISTENING
    await sm.fire(Event.UTTERANCE_COMPLETE)
    assert sm.state is State.PROCESSING
    await sm.fire(Event.LLM_RESPONSE_READY)
    assert sm.state is State.RESPONDING
    await sm.fire(Event.RESPONSE_DELIVERED)
    assert sm.state is State.LISTENING
    await sm.fire(Event.UTTERANCE_COMPLETE)
    await sm.fire(Event.LLM_RESPONSE_READY)
    await sm.fire(Event.HANGUP)
    assert sm.state is State.ENDED
    assert sm.is_terminal


@pytest.mark.asyncio
async def test_interruption_returns_to_listening() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.UTTERANCE_COMPLETE)
    await sm.fire(Event.LLM_RESPONSE_READY)
    assert sm.state is State.RESPONDING
    await sm.fire(Event.INTERRUPTED)
    assert sm.state is State.LISTENING


@pytest.mark.asyncio
async def test_escalation_path() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.UTTERANCE_COMPLETE)
    await sm.fire(Event.LLM_RESPONSE_READY)
    await sm.fire(Event.ESCALATION_REQUESTED)
    assert sm.state is State.ESCALATING
    await sm.fire(Event.ESCALATION_COMPLETE)
    assert sm.state is State.ENDED


@pytest.mark.asyncio
async def test_silence_timeout_re_prompts() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.SILENCE_TIMEOUT)
    assert sm.state is State.RESPONDING


@pytest.mark.asyncio
async def test_extended_silence_ends_call() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.EXTENDED_SILENCE)
    assert sm.state is State.ENDED


@pytest.mark.asyncio
async def test_invalid_transition_raises() -> None:
    sm = AgentStateMachine()
    with pytest.raises(InvalidTransition):
        await sm.fire(Event.LLM_RESPONSE_READY)  # not valid from IDLE


@pytest.mark.asyncio
async def test_can_handle_returns_false_for_invalid() -> None:
    sm = AgentStateMachine()
    assert sm.can_handle(Event.CALL_CONNECTED) is True
    assert sm.can_handle(Event.LLM_RESPONSE_READY) is False


@pytest.mark.asyncio
async def test_fire_if_possible_no_op_for_invalid() -> None:
    sm = AgentStateMachine()
    result = await sm.fire_if_possible(Event.LLM_RESPONSE_READY)
    assert result is None
    assert sm.state is State.IDLE


@pytest.mark.asyncio
async def test_max_duration_terminates_from_any_state() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.UTTERANCE_COMPLETE)
    assert sm.state is State.PROCESSING
    await sm.fire(Event.MAX_DURATION_REACHED)
    assert sm.state is State.ENDED


@pytest.mark.asyncio
async def test_hangup_terminates_from_listening() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.HANGUP)
    assert sm.state is State.ENDED


@pytest.mark.asyncio
async def test_listeners_called_in_order() -> None:
    seen: list[tuple[str, str]] = []
    sm = AgentStateMachine()

    async def l1(rec):
        seen.append(("l1", rec.event.value))

    async def l2(rec):
        seen.append(("l2", rec.event.value))

    sm.add_listener(l1)
    sm.add_listener(l2)
    await sm.fire(Event.CALL_CONNECTED)
    assert seen == [("l1", "call_connected"), ("l2", "call_connected")]


@pytest.mark.asyncio
async def test_history_records_all_transitions() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.UTTERANCE_COMPLETE)
    await sm.fire(Event.LLM_RESPONSE_READY)
    await sm.fire(Event.HANGUP)
    h = sm.history
    assert [r.event for r in h] == [
        Event.CALL_CONNECTED,
        Event.UTTERANCE_COMPLETE,
        Event.LLM_RESPONSE_READY,
        Event.HANGUP,
    ]
    assert h[-1].to_state is State.ENDED


@pytest.mark.asyncio
async def test_cannot_fire_after_ended() -> None:
    sm = AgentStateMachine()
    await sm.fire(Event.CALL_CONNECTED)
    await sm.fire(Event.HANGUP)
    with pytest.raises(InvalidTransition):
        await sm.fire(Event.CALL_CONNECTED)
