from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.campaign import Campaign, Lead
from src.models.conversation import Conversation
from src.models.tenant import Tenant, TenantApiKey, TenantPhoneNumber


@pytest.mark.asyncio
async def test_tenant_round_trip(test_session) -> None:
    t = Tenant(id="t_acme", slug="acme", name="Acme Telecom")
    test_session.add(t)
    await test_session.commit()
    fetched = (await test_session.execute(select(Tenant).where(Tenant.slug == "acme"))).scalar_one()
    assert fetched.name == "Acme Telecom"
    assert fetched.status == "active"


@pytest.mark.asyncio
async def test_tenant_phone_number_cascade(test_session) -> None:
    t = Tenant(id="t_acme", slug="acme", name="Acme")
    t.phone_numbers.append(TenantPhoneNumber(phone_number="+919999999999"))
    test_session.add(t)
    await test_session.commit()

    fetched = (await test_session.execute(
        select(Tenant).where(Tenant.id == "t_acme").options(selectinload(Tenant.phone_numbers))
    )).scalar_one()
    assert len(fetched.phone_numbers) == 1
    assert fetched.phone_numbers[0].provider == "twilio"


@pytest.mark.asyncio
async def test_tenant_api_key_unique_label(test_session) -> None:
    t = Tenant(id="t_acme", slug="acme", name="Acme")
    test_session.add(t)
    await test_session.commit()
    test_session.add(TenantApiKey(token_hash="h" * 64, tenant_id="t_acme", label="prod"))
    await test_session.commit()
    test_session.add(TenantApiKey(token_hash="g" * 64, tenant_id="t_acme", label="prod"))
    with pytest.raises(Exception):
        await test_session.commit()
    await test_session.rollback()


@pytest.mark.asyncio
async def test_campaign_requires_tenant_id(test_session) -> None:
    t = Tenant(id="t_acme", slug="acme", name="Acme")
    test_session.add(t)
    await test_session.commit()
    camp = Campaign(id="c1", tenant_id="t_acme", name="Plan B", config_yaml="x: 1")
    lead = Lead(id="l1", tenant_id="t_acme", campaign_id="c1", phone_number="+919999999999")
    test_session.add_all([camp, lead])
    await test_session.commit()
    fetched_camp = (await test_session.execute(select(Campaign))).scalar_one()
    assert fetched_camp.tenant_id == "t_acme"
    fetched_lead = (await test_session.execute(select(Lead))).scalar_one()
    assert fetched_lead.tenant_id == "t_acme"


@pytest.mark.asyncio
async def test_conversation_requires_tenant_id(test_session) -> None:
    t = Tenant(id="t_acme", slug="acme", name="Acme")
    test_session.add(t)
    await test_session.commit()
    conv = Conversation(
        id="conv_1", tenant_id="t_acme",
        agent_type="voicebot", channel="phone", status="active",
        pipeline_config={"stt": "sarvam"},
    )
    test_session.add(conv)
    await test_session.commit()
    fetched = (await test_session.execute(select(Conversation))).scalar_one()
    assert fetched.tenant_id == "t_acme"
