"""Campaign domain models (pydantic).

Distinct from the SQLAlchemy ORM in ``src/models/campaign.py``: those are
the persisted shape, these are the API + business-logic shape. The two
share the same field semantics but live in different namespaces so we can
evolve the API surface without rippling through migrations.

Lead status state machine (PRD §6.1):
    pending  -> in_flight -> {completed, retry, dnd}
    retry    -> in_flight (after retry_interval_hours)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# --- Enums ---------------------------------------------------------------


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class LeadStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    RETRY = "retry"
    DND = "dnd"
    FAILED = "failed"


class CallDisposition(str, Enum):
    INTERESTED_CALLBACK = "interested_callback"
    INTERESTED_TRANSFER = "interested_transfer"
    NOT_INTERESTED = "not_interested"
    BUSY_RETRY = "busy_retry"
    DND_REQUESTED = "dnd_requested"
    WRONG_NUMBER = "wrong_number"
    VOICEMAIL = "voicemail"


# --- Models --------------------------------------------------------------


class Lead(BaseModel):
    id: str
    tenant_id: str
    campaign_id: Optional[str] = None
    phone_number: str = Field(min_length=1)
    name: Optional[str] = None
    language_pref: Optional[str] = None
    crm_lead_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: LeadStatus = LeadStatus.PENDING
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("phone_number")
    @classmethod
    def _strip_phone(cls, v: str) -> str:
        return v.strip()


class Campaign(BaseModel):
    id: str
    tenant_id: str
    name: str = Field(min_length=1, max_length=255)
    status: CampaignStatus = CampaignStatus.DRAFT
    config_yaml: str = ""  # serialized campaign YAML (PRD §5.2)
    total_leads: int = 0
    calls_attempted: int = 0
    calls_answered: int = 0
    leads_qualified: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CallResult(BaseModel):
    """Outcome of one completed call. Fed back into CRM + event bus."""

    session_id: str
    tenant_id: str
    campaign_id: str
    lead_id: str
    disposition: CallDisposition
    interest_level: Optional[str] = None
    slots: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    total_turns: int = 0
    sentiment_history: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Lead import --------------------------------------------------------


class LeadImportError(ValueError):
    """Raised when a CSV row cannot be turned into a Lead."""


def parse_leads_csv(
    data: bytes,
    campaign_id: str,
    tenant_id: str,
    id_prefix: str = "lead",
) -> tuple[list[Lead], list[tuple[int, str]]]:
    """Parse a CSV blob into ``Lead`` objects.

    Required columns: ``phone_number``. Optional: ``name``, ``language_pref``,
    ``crm_lead_id``, ``id``. Any other columns land in ``metadata``.

    Returns ``(leads, errors)`` where ``errors`` is ``[(row_number, reason)]``.
    Row numbering is 1-based, matching what a spreadsheet user expects.
    """
    leads: list[Lead] = []
    errors: list[tuple[int, str]] = []
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "phone_number" not in (reader.fieldnames or []):
        raise LeadImportError("CSV must have a 'phone_number' column")

    seen_ids: set[str] = set()
    for row_num, row in enumerate(reader, start=2):  # row 1 is header
        try:
            phone = (row.get("phone_number") or "").strip()
            if not phone:
                errors.append((row_num, "missing phone_number"))
                continue
            lead_id = (row.get("id") or f"{id_prefix}_{campaign_id}_{row_num - 1:06d}").strip()
            if lead_id in seen_ids:
                errors.append((row_num, f"duplicate id '{lead_id}'"))
                continue
            seen_ids.add(lead_id)

            standard_fields = {"id", "phone_number", "name", "language_pref", "crm_lead_id"}
            metadata = {k: v for k, v in row.items() if k not in standard_fields and v}

            leads.append(Lead(
                id=lead_id,
                tenant_id=tenant_id,
                campaign_id=campaign_id,
                phone_number=phone,
                name=(row.get("name") or "").strip() or None,
                language_pref=(row.get("language_pref") or "").strip() or None,
                crm_lead_id=(row.get("crm_lead_id") or "").strip() or None,
                metadata=metadata,
            ))
        except Exception as e:  # noqa: BLE001
            errors.append((row_num, str(e)))
    return leads, errors


def leads_from_dicts(
    rows: list[dict[str, Any]],
    campaign_id: str,
    tenant_id: str,
    id_prefix: str = "lead",
) -> tuple[list[Lead], list[tuple[int, str]]]:
    """Build leads from a list of dicts (e.g. from a CRM API response)."""
    leads: list[Lead] = []
    errors: list[tuple[int, str]] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        try:
            phone = str(row.get("phone_number") or "").strip()
            if not phone:
                errors.append((i, "missing phone_number"))
                continue
            lead_id = str(row.get("id") or f"{id_prefix}_{campaign_id}_{i:06d}")
            if lead_id in seen_ids:
                errors.append((i, f"duplicate id '{lead_id}'"))
                continue
            seen_ids.add(lead_id)
            standard_fields = {"id", "phone_number", "name", "language_pref", "crm_lead_id"}
            metadata = {k: v for k, v in row.items() if k not in standard_fields and v}
            leads.append(Lead(
                id=lead_id,
                tenant_id=tenant_id,
                campaign_id=campaign_id,
                phone_number=phone,
                name=row.get("name"),
                language_pref=row.get("language_pref"),
                crm_lead_id=row.get("crm_lead_id"),
                metadata=metadata,
            ))
        except Exception as e:  # noqa: BLE001
            errors.append((i, str(e)))
    return leads, errors
