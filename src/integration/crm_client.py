"""CRM integration client.

Phase 5 ships an abstract ``ICRMClient`` plus an in-memory ``FakeCRMClient``
that mirrors the surface a real adapter would expose. Production CRMs
(Salesforce, HubSpot, custom REST) plug in by implementing the same
interface — same pattern as STT / LLM / TTS.

Methods are intentionally narrow:
- ``fetch_leads(campaign_id)``  pull a fresh lead list (PRD: lead_list_source=crm)
- ``update_lead(call_result)``  push call disposition + slots back to CRM
- ``mark_dnd(phone_number)``    propagate DND requests upstream
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from src.campaign.models import CallResult, Lead, leads_from_dicts

log = logging.getLogger(__name__)


class ICRMClient(Protocol):
    async def fetch_leads(self, campaign_id: str, tenant_id: str) -> list[Lead]: ...

    async def update_lead(self, call_result: CallResult) -> None: ...

    async def mark_dnd(self, phone_number: str) -> None: ...


@dataclass
class FakeCRMClient:
    """In-memory CRM stand-in. Captures every interaction so tests can
    assert on the side effects without spinning up an external system."""

    seed_leads: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    updates: list[CallResult] = field(default_factory=list)
    dnd_requests: list[str] = field(default_factory=list)

    async def fetch_leads(self, campaign_id: str, tenant_id: str = "t_default") -> list[Lead]:
        rows = self.seed_leads.get(campaign_id, [])
        leads, errors = leads_from_dicts(
            rows, campaign_id=campaign_id, tenant_id=tenant_id, id_prefix="crm"
        )
        if errors:
            log.warning("crm leads import had errors", extra={"errors": errors})
        return leads

    async def update_lead(self, call_result: CallResult) -> None:
        self.updates.append(call_result)

    async def mark_dnd(self, phone_number: str) -> None:
        self.dnd_requests.append(phone_number)


# --- WhatsApp / chat handoff --------------------------------------------


class IChatChannel(Protocol):
    """Outbound messaging channel — WhatsApp Business API, Telegram, etc."""

    async def send_message(self, to_number: str, text: str, language: Optional[str] = None) -> str: ...


@dataclass
class FakeChatChannel:
    """In-memory channel — captures sent messages for assertions."""

    sent: list[dict[str, Any]] = field(default_factory=list)

    async def send_message(self, to_number: str, text: str, language: Optional[str] = None) -> str:
        msg_id = f"msg_{len(self.sent) + 1:05d}"
        self.sent.append({
            "id": msg_id,
            "to": to_number,
            "text": text,
            "language": language,
        })
        return msg_id
