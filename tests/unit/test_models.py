from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.campaign import Campaign, Lead
from src.models.conversation import Conversation, Event, Turn
from src.models.tenant import Tenant


@pytest.mark.asyncio
async def test_campaign_lead_round_trip(test_session) -> None:
    test_session.add(Tenant(id="t1", slug="t1", name="T1"))
    await test_session.commit()
    camp = Campaign(id="camp_1", tenant_id="t1", name="Plan B", config_yaml="campaign: {}")
    lead = Lead(id="lead_1", tenant_id="t1", campaign_id="camp_1",
                phone_number="+919999999999", name="A")
    test_session.add_all([camp, lead])
    await test_session.commit()

    fetched = (
        await test_session.execute(
            select(Campaign).options(selectinload(Campaign.leads))
        )
    ).scalars().all()
    assert len(fetched) == 1
    assert fetched[0].name == "Plan B"
    assert fetched[0].leads[0].phone_number == "+919999999999"


@pytest.mark.asyncio
async def test_conversation_with_turns_and_events(test_session) -> None:
    test_session.add(Tenant(id="t1", slug="t1", name="T1"))
    await test_session.commit()
    conv = Conversation(
        id="conv_1",
        tenant_id="t1",
        agent_type="voicebot",
        channel="phone",
        status="active",
        pipeline_config={"stt": {"provider": "sarvam"}},
    )
    conv.turns.append(
        Turn(turn_number=1, role="agent", content="Namaste", language="hi")
    )
    conv.events.append(Event(event_type="call.answered", payload={"x": 1}))
    test_session.add(conv)
    await test_session.commit()

    fetched = (
        await test_session.execute(
            select(Conversation)
            .where(Conversation.id == "conv_1")
            .options(selectinload(Conversation.turns), selectinload(Conversation.events))
        )
    ).scalar_one()
    assert len(fetched.turns) == 1
    assert fetched.turns[0].content == "Namaste"
    assert fetched.events[0].event_type == "call.answered"
