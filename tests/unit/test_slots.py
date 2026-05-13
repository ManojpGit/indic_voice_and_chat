from __future__ import annotations

import yaml

from src.dialogue.slots import SlotFiller, SlotSchema, SlotType


SAMPLE_SCHEMA_YAML = """
lead_name:        { type: string,   required: true,  source: crm }
interest_level:   { type: enum,     required: true,  values: [hot, warm, cold, not_interested] }
budget_range:     { type: string,   required: false }
callback_time:    { type: datetime, required: false }
whatsapp_number:  { type: phone,    required: false }
age:              { type: number,   required: false }
opted_in:         { type: boolean,  required: false }
"""


def _schema() -> SlotSchema:
    return SlotSchema.from_campaign_yaml(yaml.safe_load(SAMPLE_SCHEMA_YAML))


def test_schema_round_trip() -> None:
    s = _schema()
    assert s.specs["lead_name"].type is SlotType.STRING
    assert s.specs["lead_name"].required is True
    assert s.specs["interest_level"].values == ["hot", "warm", "cold", "not_interested"]
    assert sorted(s.required_names()) == ["interest_level", "lead_name"]


def test_apply_string_slot() -> None:
    f = SlotFiller(_schema())
    applied = f.apply_updates({"lead_name": "Manoj"})
    assert applied == {"lead_name": "Manoj"}
    assert f.get("lead_name") == "Manoj"


def test_apply_enum_validation() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"interest_level": "hot"})
    assert f.get("interest_level") == "hot"

    f.apply_updates({"interest_level": "lukewarm"})  # not allowed
    assert f.get("interest_level") == "hot"  # unchanged
    assert any(reason for _, _, reason in f.rejected if "allowed values" in reason)


def test_apply_phone_validation() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"whatsapp_number": "+91 9999 999 999"})
    assert f.get("whatsapp_number") == "+91 9999 999 999"
    f.apply_updates({"whatsapp_number": "not a phone"})
    assert f.get("whatsapp_number") == "+91 9999 999 999"


def test_apply_datetime_iso() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"callback_time": "2026-06-12T15:30:00"})
    assert f.get("callback_time") == "2026-06-12T15:30:00"


def test_apply_datetime_opaque_string_is_kept() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"callback_time": "tomorrow at 3pm"})
    assert f.get("callback_time") == "tomorrow at 3pm"


def test_apply_number_coerces() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"age": "42"})
    assert f.get("age") == 42.0


def test_apply_boolean_string() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"opted_in": "haan"})
    assert f.get("opted_in") is True
    f.apply_updates({"opted_in": "nahi"})
    assert f.get("opted_in") is False


def test_unknown_slot_is_rejected() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"made_up_slot": "x"})
    assert f.values == {}
    assert any(name == "made_up_slot" for name, _, _ in f.rejected)


def test_empty_value_is_skipped() -> None:
    f = SlotFiller(_schema())
    f.apply_updates({"lead_name": ""})
    assert f.values == {}
    f.apply_updates({"lead_name": None})
    assert f.values == {}


def test_completeness() -> None:
    f = SlotFiller(_schema())
    assert sorted(f.missing_required()) == ["interest_level", "lead_name"]
    assert not f.is_complete()
    f.apply_updates({"lead_name": "M", "interest_level": "warm"})
    assert f.is_complete()
    assert f.missing_required() == []
