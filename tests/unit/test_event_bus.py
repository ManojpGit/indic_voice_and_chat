from __future__ import annotations

import asyncio

import pytest

from src.integration.event_bus import (
    Event,
    EventBus,
    EventType,
    emit_call_completed,
    emit_lead_qualified,
)


@pytest.mark.asyncio
async def test_subscribe_and_publish_exact_match() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(e: Event) -> None:
        received.append(e)

    bus.subscribe(EventType.CALL_COMPLETED, handler)
    await bus.publish(Event(type=EventType.CALL_COMPLETED, payload={"x": 1}))
    assert len(received) == 1
    assert received[0].payload["x"] == 1


@pytest.mark.asyncio
async def test_wildcard_subscriber_receives_all() -> None:
    bus = EventBus()
    received: list[str] = []

    async def all_handler(e: Event) -> None:
        received.append(e.type)

    bus.subscribe("*", all_handler)
    await bus.publish(Event(type="call.initiated"))
    await bus.publish(Event(type="lead.qualified"))
    await bus.publish(Event(type="anything.else"))
    assert received == ["call.initiated", "lead.qualified", "anything.else"]


@pytest.mark.asyncio
async def test_no_subscribers_publish_is_noop() -> None:
    bus = EventBus()
    await bus.publish(Event(type="orphan"))


@pytest.mark.asyncio
async def test_handler_error_does_not_break_other_handlers() -> None:
    bus = EventBus()
    received: list[str] = []

    async def bad(e: Event) -> None:
        raise RuntimeError("boom")

    async def good(e: Event) -> None:
        received.append(e.type)

    bus.subscribe(EventType.CALL_COMPLETED, bad)
    bus.subscribe(EventType.CALL_COMPLETED, good)
    await bus.publish(Event(type=EventType.CALL_COMPLETED))
    assert received == [EventType.CALL_COMPLETED]


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def h(e: Event) -> None:
        received.append(e)

    bus.subscribe("test.x", h)
    await bus.publish(Event(type="test.x"))
    bus.unsubscribe("test.x", h)
    await bus.publish(Event(type="test.x"))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_handlers_run_concurrently() -> None:
    bus = EventBus()
    order: list[str] = []
    finish = asyncio.Event()

    async def slow(e: Event) -> None:
        order.append("slow_start")
        await asyncio.sleep(0.05)
        order.append("slow_end")
        finish.set()

    async def fast(e: Event) -> None:
        order.append("fast")

    bus.subscribe("evt", slow)
    bus.subscribe("evt", fast)
    await bus.publish(Event(type="evt"))
    # Both ran; "fast" landed before "slow_end" because they ran concurrently
    assert "fast" in order
    assert order.index("fast") < order.index("slow_end")


@pytest.mark.asyncio
async def test_emit_call_completed_helper() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def h(e: Event) -> None:
        seen.append(e)

    bus.subscribe(EventType.CALL_COMPLETED, h)
    await emit_call_completed(
        bus,
        tenant_id="t1",
        session_id="s1",
        campaign_id="c1",
        lead_id="l1",
        disposition="interested_callback",
        duration_ms=12345,
    )
    assert seen[0].payload == {
        "tenant_id": "t1",
        "session_id": "s1",
        "campaign_id": "c1",
        "lead_id": "l1",
        "disposition": "interested_callback",
        "duration_ms": 12345,
    }


@pytest.mark.asyncio
async def test_emit_lead_qualified_helper() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def h(e: Event) -> None:
        seen.append(e)

    bus.subscribe(EventType.LEAD_QUALIFIED, h)
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="hot",
        slots={"whatsapp_number": "+919999"},
    )
    assert seen[0].payload["interest_level"] == "hot"
    assert seen[0].payload["slots"]["whatsapp_number"] == "+919999"
