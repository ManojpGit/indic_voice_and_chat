from __future__ import annotations

from typing import Any

import pytest

from src.integration.event_bus import Event, EventBus
from src.integration.webhooks import (
    WebhookConfig,
    WebhookManager,
    WebhookRegistration,
)


# --- Pattern matching ---------------------------------------------------


def test_registration_wildcard_matches_anything() -> None:
    r = WebhookRegistration(id="x", url="http://x", event_filters=["*"])
    assert r.matches("call.initiated") is True
    assert r.matches("anything") is True


def test_registration_exact_match() -> None:
    r = WebhookRegistration(id="x", url="http://x", event_filters=["call.completed"])
    assert r.matches("call.completed") is True
    assert r.matches("call.initiated") is False


def test_registration_prefix_match() -> None:
    r = WebhookRegistration(id="x", url="http://x", event_filters=["call.*"])
    assert r.matches("call.initiated") is True
    assert r.matches("call.completed") is True
    assert r.matches("lead.qualified") is False


def test_inactive_registration_never_matches() -> None:
    r = WebhookRegistration(id="x", url="http://x", event_filters=["*"], active=False)
    assert r.matches("anything") is False


# --- Manager registration ----------------------------------------------


@pytest.mark.asyncio
async def test_register_and_list() -> None:
    m = WebhookManager()
    r1 = m.register("https://a.example/webhook", ["call.*"])
    r2 = m.register("https://b.example/webhook")
    assert {r.id for r in m.list()} == {r1.id, r2.id}
    assert r1.event_filters == ["call.*"]
    assert r2.event_filters == ["*"]


@pytest.mark.asyncio
async def test_unregister() -> None:
    m = WebhookManager()
    r = m.register("https://x")
    assert m.unregister(r.id) is True
    assert m.list() == []
    assert m.unregister("nonexistent") is False


# --- Delivery ----------------------------------------------------------


def _fake_poster(returns: list[int]) -> Any:
    """Return a poster that yields successive status codes from ``returns``."""
    calls: list[tuple[str, dict, float]] = []
    iter_returns = iter(returns)

    async def post(url: str, json: dict, timeout: float) -> int:
        calls.append((url, json, timeout))
        try:
            return next(iter_returns)
        except StopIteration:
            return 200

    post.calls = calls  # type: ignore[attr-defined]
    return post


@pytest.mark.asyncio
async def test_delivery_to_matching_webhook_only() -> None:
    bus = EventBus()
    poster = _fake_poster([200])
    m = WebhookManager(bus=bus, http_post=poster)
    matching = m.register("https://match.example", ["call.*"])
    other = m.register("https://other.example", ["lead.qualified"])

    await bus.publish(Event(type="call.initiated", payload={"x": 1}))
    assert len(poster.calls) == 1
    assert poster.calls[0][0] == matching.url
    assert poster.calls[0][1]["event_type"] == "call.initiated"
    assert poster.calls[0][1]["payload"] == {"x": 1}
    # ``other`` got nothing
    assert all(c[0] != other.url for c in poster.calls)


@pytest.mark.asyncio
async def test_delivery_retries_on_failure() -> None:
    bus = EventBus()
    poster = _fake_poster([500, 500, 200])  # third try succeeds
    m = WebhookManager(
        bus=bus,
        http_post=poster,
        config=WebhookConfig(timeout_s=1, max_attempts=3, backoff_base_s=0.0),
    )
    m.register("https://x", ["*"])
    await bus.publish(Event(type="any"))
    assert len(poster.calls) == 3
    assert m.delivered[-1][2] == 200


@pytest.mark.asyncio
async def test_delivery_records_final_failure() -> None:
    bus = EventBus()
    poster = _fake_poster([500, 500, 500])
    m = WebhookManager(
        bus=bus,
        http_post=poster,
        config=WebhookConfig(timeout_s=1, max_attempts=3, backoff_base_s=0.0),
    )
    m.register("https://x", ["*"])
    await bus.publish(Event(type="any"))
    assert m.delivered[-1][2] == -1


@pytest.mark.asyncio
async def test_no_subscribers_for_event_no_calls() -> None:
    bus = EventBus()
    poster = _fake_poster([200])
    m = WebhookManager(bus=bus, http_post=poster)
    m.register("https://x", ["call.*"])
    await bus.publish(Event(type="lead.qualified"))
    assert poster.calls == []


@pytest.mark.asyncio
async def test_payload_includes_metadata() -> None:
    bus = EventBus()
    poster = _fake_poster([200])
    m = WebhookManager(bus=bus, http_post=poster)
    m.register("https://x", ["*"])
    await bus.publish(Event(type="evt", payload={"x": 1}, source="orchestrator"))
    body = poster.calls[0][1]
    assert body["event_type"] == "evt"
    assert body["source"] == "orchestrator"
    assert body["payload"] == {"x": 1}
    assert "occurred_at" in body
    assert "webhook_id" in body
