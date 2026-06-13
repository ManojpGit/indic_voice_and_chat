"""End-to-end campaign flow.

Wires every Phase 5 piece together against a fake telephony layer, driving the
``CampaignOrchestrator`` directly (the HTTP auto-dial endpoints were retired in
the API/DB restructure — calls are now placed per-lead via Call Lead):

    leads CSV -> parse_leads_csv
    orchestrator.run(campaign, leads)
        -> orchestrator picks leads via scheduler
        -> dispatch returns CallResult per lead
        -> CRM.update_lead recorded
        -> EventBus emits call.initiated, call.completed, lead.qualified
        -> WebhookManager fans out to registered URLs
        -> WhatsAppHandoff sends follow-up on lead.qualified

Asserts the full causal chain: ingest -> dispatch -> events -> webhooks ->
CRM updates -> WhatsApp handoff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.campaign.dnd_filter import CallingHoursPolicy, DNDFilter, InMemoryDNDStore
from src.campaign.models import (
    CallDisposition,
    CallResult,
    Lead,
    parse_leads_csv,
)
from src.campaign.models import Campaign as CampaignModel
from src.campaign.orchestrator import CampaignOrchestrator
from src.campaign.scheduler import CallScheduler, RateLimitConfig
from src.integration.crm_client import FakeChatChannel, FakeCRMClient
from src.integration.event_bus import EventBus
from src.integration.handoff import WhatsAppHandoff
from src.integration.webhooks import WebhookConfig, WebhookManager


# Per-lead disposition table for the fake dispatcher.
DISPOSITION_BY_PHONE: dict[str, dict[str, Any]] = {
    "+919999999991": {
        "disposition": CallDisposition.INTERESTED_CALLBACK,
        "interest_level": "hot",
        "slots": {"interest_level": "hot", "whatsapp_number": "+919999999991"},
    },
    "+919999999992": {
        "disposition": CallDisposition.NOT_INTERESTED,
        "interest_level": "cold",
        "slots": {"interest_level": "cold"},
    },
    "+919999999993": {
        "disposition": CallDisposition.BUSY_RETRY,
        "interest_level": None,
        "slots": {},
    },
    "+919999999994": {
        "disposition": CallDisposition.DND_REQUESTED,
        "interest_level": None,
        "slots": {},
    },
}


async def _zero_sleep(seconds: float) -> None:
    """Drive the orchestrator loop without real delays (tests)."""
    return None


def _make_dispatch():
    async def dispatch(camp: CampaignModel, lead: Lead) -> CallResult:
        spec = DISPOSITION_BY_PHONE.get(lead.phone_number, {
            "disposition": CallDisposition.NOT_INTERESTED,
            "interest_level": None,
            "slots": {},
        })
        now = datetime.utcnow()
        return CallResult(
            session_id=f"sess-{lead.id}",
            tenant_id=camp.tenant_id,
            campaign_id=camp.id,
            lead_id=lead.id,
            disposition=spec["disposition"],
            interest_level=spec["interest_level"],
            slots=spec["slots"],
            duration_ms=8000,
            started_at=now,
            ended_at=now,
        )

    return dispatch


@pytest.fixture
def wired():
    bus = EventBus()
    crm = FakeCRMClient()
    chat_channel = FakeChatChannel()
    handoff = WhatsAppHandoff(bus, chat_channel)

    # Webhooks with a fake poster so we don't actually hit the network.
    webhook_calls: list[tuple[str, dict, float]] = []

    async def fake_post(url: str, json: dict, timeout: float) -> int:
        webhook_calls.append((url, json, timeout))
        return 200

    webhook_manager = WebhookManager(
        bus=bus,
        http_post=fake_post,
        config=WebhookConfig(timeout_s=1, max_attempts=1, backoff_base_s=0.0),
    )

    sched = CallScheduler(
        hours=CallingHoursPolicy(start="00:00", end="23:59", skip_weekday=None),
        dnd_filter=DNDFilter(InMemoryDNDStore()),
        rate_limit=RateLimitConfig(calls_per_minute=100, max_concurrent_calls=4),
    )
    orchestrator = CampaignOrchestrator(
        scheduler=sched,
        dispatch=_make_dispatch(),
        bus=bus,
        crm=crm,
        max_concurrent=4,
    )

    return {
        "orchestrator": orchestrator,
        "bus": bus,
        "crm": crm,
        "chat_channel": chat_channel,
        "handoff": handoff,
        "webhook_calls": webhook_calls,
        "webhook_manager": webhook_manager,
    }


@pytest.mark.asyncio
async def test_full_campaign_flow(wired) -> None:
    orchestrator: CampaignOrchestrator = wired["orchestrator"]
    crm: FakeCRMClient = wired["crm"]
    chat_channel: FakeChatChannel = wired["chat_channel"]
    webhook_calls = wired["webhook_calls"]
    webhook_manager: WebhookManager = wired["webhook_manager"]

    # 1. Register webhook for all events
    reg = webhook_manager.register(url="https://example.com/wh", event_filters=["*"])
    assert reg.id

    # 2. Create campaign + 3. ingest leads CSV (parsed the same way the API does)
    campaign = CampaignModel(id="c1", tenant_id="t1", name="Plan B Launch")
    csv = (
        b"phone_number,name\n"
        b"+919999999991,Manoj\n"     # interested + whatsapp -> handoff
        b"+919999999992,Aarti\n"     # not interested
        b"+919999999993,Sanjay\n"    # busy -> retry
        b"+919999999994,Kavita\n"    # dnd
    )
    leads, errors = parse_leads_csv(csv, campaign_id="c1", tenant_id="t1")
    assert errors == []
    assert len(leads) == 4

    # 4. Run the campaign. Bounded iterations + zero-sleep keep it deterministic:
    # the busy-retry lead (+...993) stays in RETRY (its next_retry_at is minutes
    # out and never becomes due in this fast loop), so each lead is dispatched
    # exactly once.
    run = await orchestrator.run(
        campaign, leads, max_iterations=50, sleep_fn=_zero_sleep
    )

    assert run.campaign.calls_attempted == 4
    # 'answered' counts everything except VOICEMAIL.
    assert run.campaign.calls_answered == 4
    assert run.campaign.leads_qualified == 1

    # 5. CRM updates: one row per dispatch (4 total)
    assert len(crm.updates) == 4
    dispositions = {u.lead_id: u.disposition for u in crm.updates}
    assert any(d == CallDisposition.INTERESTED_CALLBACK for d in dispositions.values())
    assert any(d == CallDisposition.DND_REQUESTED for d in dispositions.values())
    # DND request was forwarded
    assert crm.dnd_requests == ["+919999999994"]

    # 6. WhatsApp handoff fired exactly for the qualifying lead
    assert len(chat_channel.sent) == 1
    assert chat_channel.sent[0]["to"] == "+919999999991"

    # 7. Webhook events delivered
    event_types = [c[1]["event_type"] for c in webhook_calls]
    assert event_types.count("call.initiated") == 4
    assert event_types.count("call.completed") == 4
    assert event_types.count("lead.qualified") == 1

    # 8. Final run stats mirror the campaign counters.
    assert run.stats["calls_attempted"] == 4
    assert run.stats["leads_qualified"] == 1
