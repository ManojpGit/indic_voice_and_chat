"""Unit tests for the call-record persistence helpers."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api.call_store import (
    compute_call_cost,
    count_active_calls,
    deliver_to_persister,
    record_outcome,
    set_call_outcome_persister,
)
from src.models.database import Base
from src.models.conversation import Conversation
from src.models.tenant import ProviderCost, Tenant


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        # catalog: layered providers + telephony + s2s
        s.add_all([
            ProviderCost(kind="stt", provider="groq", cost_per_min=0.01),
            ProviderCost(kind="llm", provider="gemini", cost_per_min=0.02),
            ProviderCost(kind="tts", provider="sarvam", cost_per_min=0.03),
            ProviderCost(kind="telephony", provider="twilio", cost_per_min=0.10),
            ProviderCost(kind="s2s", provider="gemini_live", cost_per_min=0.50),
        ])
        await s.commit()
    yield maker
    await engine.dispose()


def _conv(**over):
    base = dict(
        id="call_1", tenant_id="t1", agent_type="voicebot", channel="voice",
        status="in_progress", pipeline_config={}, provider_call_sid="SID-1",
        mode="layered", stt_provider="groq", llm_provider="gemini",
        tts_provider="sarvam", telephony_provider="twilio",
    )
    base.update(over)
    return Conversation(**base)


async def test_compute_cost_layered(sm):
    async with sm() as s:
        # 2 minutes layered: (0.01+0.02+0.03+0.10)/min * 2 = 0.32
        cost = await compute_call_cost(
            s, mode="layered", stt_provider="groq", llm_provider="gemini",
            tts_provider="sarvam", telephony_provider="twilio", duration_ms=120_000)
    assert cost == pytest.approx(0.32)


async def test_compute_cost_s2s(sm):
    async with sm() as s:
        # 1 minute s2s: (0.50 + 0.10) = 0.60
        cost = await compute_call_cost(
            s, mode="s2s", realtime_provider="gemini_live",
            telephony_provider="twilio", duration_ms=60_000)
    assert cost == pytest.approx(0.60)


async def test_compute_cost_zero_duration(sm):
    async with sm() as s:
        assert await compute_call_cost(
            s, mode="layered", telephony_provider="twilio", duration_ms=0) == 0.0


async def test_compute_cost_unknown_provider_skipped(sm):
    async with sm() as s:
        # deepgram not in catalog -> contributes 0; telephony twilio counts.
        cost = await compute_call_cost(
            s, mode="layered", stt_provider="deepgram", telephony_provider="twilio",
            duration_ms=60_000)
    assert cost == pytest.approx(0.10)


async def test_count_active_calls(sm):
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="s1", status="in_progress"))
        s.add(_conv(id="c2", provider_call_sid="s2", status="answered"))
        s.add(_conv(id="c3", provider_call_sid="s3", status="ended"))
        await s.commit()
        assert await count_active_calls(s, "t1") == 2


async def test_record_outcome_writes_and_computes_cost(sm):
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="SID-9"))
        await s.commit()
    async with sm() as s:
        row = await record_outcome(
            s, "SID-9", outcome="interested", summary="Wants a callback",
            notes="Prefers evenings", duration_ms=120_000)
    assert row is not None
    assert row.status == "ended"
    assert row.outcome == "interested"
    assert row.summary == "Wants a callback"
    assert row.duration_ms == 120_000
    assert row.cost == pytest.approx(0.32)
    assert row.ended_at is not None


async def test_record_outcome_unknown_sid_returns_none(sm):
    async with sm() as s:
        assert await record_outcome(s, "missing-sid", outcome="x") is None


async def test_deliver_to_persister_writes_outcome(sm):
    """Simulates a bridge teardown handing its outcome payload to the wired
    persister, which writes the conversations row (the main.py wiring shape)."""
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="SID-P"))
        await s.commit()

    async def _persister(call_sid, payload):
        async with sm() as s:
            await record_outcome(
                s, call_sid, outcome=payload.get("outcome"),
                summary=payload.get("summary"), notes=payload.get("notes"))

    set_call_outcome_persister(_persister)
    try:
        await deliver_to_persister("SID-P", {
            "outcome": "interested", "summary": "Wants a callback", "notes": "Evenings"})
    finally:
        set_call_outcome_persister(None)

    async with sm() as s:
        from sqlalchemy import select
        got = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "SID-P")
        )).scalar_one()
    assert got.status == "ended"
    assert got.outcome == "interested"
    assert got.summary == "Wants a callback"
    # Cost derived (duration from started_at -> ended_at; tiny but >= 0).
    assert got.cost is not None and got.cost >= 0.0


async def test_deliver_to_persister_noop_without_persister(sm):
    # Unset persister + missing sid must both be safe no-ops (never raise).
    set_call_outcome_persister(None)
    await deliver_to_persister("any-sid", {"outcome": "x"})
    await deliver_to_persister(None, {"outcome": "x"})


async def test_deliver_to_persister_swallows_errors(sm):
    async def _boom(call_sid, payload):
        raise RuntimeError("db down")

    set_call_outcome_persister(_boom)
    try:
        # Must not propagate — teardown has to survive a failing persister.
        await deliver_to_persister("sid", {"outcome": "x"})
    finally:
        set_call_outcome_persister(None)
