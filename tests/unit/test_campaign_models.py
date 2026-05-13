from __future__ import annotations

import pytest

from src.campaign.models import (
    Campaign,
    CampaignStatus,
    Lead,
    LeadImportError,
    LeadStatus,
    leads_from_dicts,
    parse_leads_csv,
)


def test_campaign_defaults() -> None:
    c = Campaign(id="c1", tenant_id="t1", name="Plan B Launch")
    assert c.status is CampaignStatus.DRAFT
    assert c.calls_attempted == 0
    assert c.calls_answered == 0


def test_lead_strips_phone() -> None:
    lead = Lead(id="l1", tenant_id="t1", phone_number="  +91 9999999999  ")
    assert lead.phone_number == "+91 9999999999"


def test_lead_default_status() -> None:
    lead = Lead(id="l1", tenant_id="t1", phone_number="+919999999999")
    assert lead.status is LeadStatus.PENDING


# --- CSV import ---------------------------------------------------------


def test_parse_csv_minimal() -> None:
    csv_bytes = b"phone_number\n+919999999999\n+918888888888\n"
    leads, errors = parse_leads_csv(csv_bytes, campaign_id="c1", tenant_id="t1")
    assert errors == []
    assert len(leads) == 2
    assert leads[0].phone_number == "+919999999999"
    assert leads[0].id == "lead_c1_000001"


def test_parse_csv_full_columns() -> None:
    csv_bytes = (
        b"id,phone_number,name,language_pref,crm_lead_id,city\n"
        b"l1,+919999999999,Manoj,hi,CRM-42,Pune\n"
    )
    leads, errors = parse_leads_csv(csv_bytes, campaign_id="c1", tenant_id="t1")
    assert errors == []
    assert len(leads) == 1
    assert leads[0].id == "l1"
    assert leads[0].name == "Manoj"
    assert leads[0].language_pref == "hi"
    assert leads[0].crm_lead_id == "CRM-42"
    assert leads[0].metadata == {"city": "Pune"}


def test_parse_csv_missing_phone_column_raises() -> None:
    with pytest.raises(LeadImportError):
        parse_leads_csv(b"name\nManoj\n", campaign_id="c1", tenant_id="t1")


def test_parse_csv_skips_empty_phone() -> None:
    csv_bytes = b"phone_number,name\n+919999999999,A\n,B\n"
    leads, errors = parse_leads_csv(csv_bytes, campaign_id="c1", tenant_id="t1")
    assert len(leads) == 1
    assert errors == [(3, "missing phone_number")]


def test_parse_csv_duplicate_ids_reported() -> None:
    csv_bytes = b"id,phone_number\nx,+1\nx,+2\n"
    leads, errors = parse_leads_csv(csv_bytes, campaign_id="c1", tenant_id="t1")
    assert len(leads) == 1
    assert errors == [(3, "duplicate id 'x'")]


def test_parse_csv_handles_utf8_bom() -> None:
    csv_bytes = b"\xef\xbb\xbfphone_number\n+919999999999\n"
    leads, errors = parse_leads_csv(csv_bytes, campaign_id="c1", tenant_id="t1")
    assert errors == []
    assert len(leads) == 1


# --- leads_from_dicts ---------------------------------------------------


def test_leads_from_dicts_full() -> None:
    rows = [
        {"id": "l1", "phone_number": "+91", "name": "A", "city": "Mumbai"},
        {"phone_number": "+92"},
    ]
    leads, errors = leads_from_dicts(rows, campaign_id="c1", tenant_id="t1")
    assert errors == []
    assert len(leads) == 2
    assert leads[0].metadata == {"city": "Mumbai"}
    assert leads[1].id == "lead_c1_000001"


def test_leads_from_dicts_skips_missing_phone() -> None:
    rows = [{"phone_number": "+1"}, {"name": "no phone"}]
    leads, errors = leads_from_dicts(rows, campaign_id="c1", tenant_id="t1")
    assert len(leads) == 1
    assert errors == [(1, "missing phone_number")]
