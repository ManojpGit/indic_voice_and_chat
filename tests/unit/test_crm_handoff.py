from __future__ import annotations

from datetime import datetime

import pytest

from src.campaign.models import CallDisposition, CallResult, Lead
from src.integration.crm_client import FakeChatChannel, FakeCRMClient
from src.integration.event_bus import EventBus, emit_lead_qualified
from src.integration.handoff import HandoffConfig, WhatsAppHandoff


# --- FakeCRMClient ------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_crm_fetch_leads() -> None:
    crm = FakeCRMClient(seed_leads={
        "c1": [
            {"id": "l1", "phone_number": "+91", "name": "A"},
            {"phone_number": "+92", "name": "B"},
        ]
    })
    leads = await crm.fetch_leads("c1", tenant_id="t1")
    assert len(leads) == 2
    assert leads[0].id == "l1"
    assert leads[0].name == "A"


@pytest.mark.asyncio
async def test_fake_crm_update_lead_records_call_result() -> None:
    crm = FakeCRMClient()
    cr = CallResult(
        session_id="s1",
        tenant_id="t1",
        campaign_id="c1",
        lead_id="l1",
        disposition=CallDisposition.INTERESTED_CALLBACK,
        duration_ms=12000,
        started_at=datetime.utcnow(),
        ended_at=datetime.utcnow(),
    )
    await crm.update_lead(cr)
    assert crm.updates == [cr]


@pytest.mark.asyncio
async def test_fake_crm_mark_dnd() -> None:
    crm = FakeCRMClient()
    await crm.mark_dnd("+919999999999")
    assert crm.dnd_requests == ["+919999999999"]


@pytest.mark.asyncio
async def test_fake_chat_channel_returns_id() -> None:
    ch = FakeChatChannel()
    msg_id = await ch.send_message("+919999999999", "Hi", language="en")
    assert msg_id.startswith("msg_")
    assert ch.sent == [
        {"id": msg_id, "to": "+919999999999", "text": "Hi", "language": "en"}
    ]


# --- WhatsAppHandoff ---------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_dispatches_message_for_qualifying_lead() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)

    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="hot",
        slots={"whatsapp_number": "+919999999999", "language": "hi"},
    )
    assert len(ch.sent) == 1
    assert ch.sent[0]["to"] == "+919999999999"
    assert ch.sent[0]["language"] == "hi"
    assert "Namaste" in ch.sent[0]["text"]


@pytest.mark.asyncio
async def test_handoff_uses_english_template_when_lang_en() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="hot",
        slots={"whatsapp_number": "+919999", "language": "en"},
    )
    assert "Thanks for the call" in ch.sent[0]["text"]


@pytest.mark.asyncio
async def test_handoff_dedupes_per_session() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)
    for _ in range(3):
        await emit_lead_qualified(
            bus,
            tenant_id="t1",
            session_id="s1",
            lead_id="l1",
            interest_level="hot",
            slots={"whatsapp_number": "+91"},
        )
    assert len(ch.sent) == 1


@pytest.mark.asyncio
async def test_handoff_skips_when_not_qualifying() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="not_interested",
        slots={"whatsapp_number": "+91"},
    )
    assert ch.sent == []


@pytest.mark.asyncio
async def test_handoff_skips_when_no_whatsapp_number() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="hot",
        slots={},
    )
    assert ch.sent == []


@pytest.mark.asyncio
async def test_handoff_treats_interested_callback_as_warm() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch)
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="interested_callback",
        slots={"whatsapp_number": "+91"},
    )
    assert len(ch.sent) == 1
    assert "Following up" in ch.sent[0]["text"] or "baat karke accha" in ch.sent[0]["text"]


@pytest.mark.asyncio
async def test_handoff_custom_templates() -> None:
    bus = EventBus()
    ch = FakeChatChannel()
    WhatsAppHandoff(bus, ch, HandoffConfig(
        templates={"hot": {"en": "Custom hot english template"}},
        qualifying_levels=("hot",),
        default_language="en",
    ))
    await emit_lead_qualified(
        bus,
        tenant_id="t1",
        session_id="s1",
        lead_id="l1",
        interest_level="hot",
        slots={"whatsapp_number": "+91", "language": "en"},
    )
    assert ch.sent[0]["text"] == "Custom hot english template"
