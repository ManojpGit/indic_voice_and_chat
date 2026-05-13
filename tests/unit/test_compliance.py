from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.campaign.dnd_filter import (
    IST,
    CallingHoursPolicy,
    DNDFilter,
    InMemoryDNDStore,
    normalize_phone,
)


# --- normalize_phone ----------------------------------------------------


def test_normalize_phone_strips_punctuation() -> None:
    assert normalize_phone("+91 (999) 999-9999") == "+919999999999"


def test_normalize_phone_preserves_leading_plus() -> None:
    assert normalize_phone("+91-9999999999") == "+919999999999"


def test_normalize_phone_empty() -> None:
    assert normalize_phone("") == ""
    assert normalize_phone("   ") == ""


# --- InMemoryDNDStore ---------------------------------------------------


def test_dnd_store_add_check_remove() -> None:
    s = InMemoryDNDStore()
    s.add("+91 9999 999 999")
    assert s.is_dnd("+919999999999") is True
    assert s.is_dnd("+918888888888") is False
    s.remove("+919999999999")
    assert s.is_dnd("+919999999999") is False


def test_dnd_store_initial_seed() -> None:
    s = InMemoryDNDStore(initial=["+919999999999", "+918888888888"])
    assert len(s) == 2


def test_dnd_store_normalizes_on_lookup() -> None:
    s = InMemoryDNDStore(initial=["+919999999999"])
    assert s.is_dnd("+91 (9999) 9999-99") is True


# --- DNDFilter ----------------------------------------------------------


def test_dnd_filter_blocks_listed() -> None:
    f = DNDFilter(InMemoryDNDStore(["+919999999999"]))
    assert f.is_blocked("+919999999999") is True
    assert f.is_blocked("+918888888888") is False


def test_dnd_filter_disabled_passes_through() -> None:
    f = DNDFilter(InMemoryDNDStore(["+919999999999"]), enabled=False)
    assert f.is_blocked("+919999999999") is False


def test_dnd_filter_filter_blocked_partitions_input() -> None:
    f = DNDFilter(InMemoryDNDStore(["+919999999999", "+917777777777"]))
    allowed, blocked = f.filter_blocked([
        "+919999999999",  # blocked
        "+918888888888",  # allowed
        "+917777777777",  # blocked
    ])
    assert allowed == ["+918888888888"]
    assert blocked == ["+919999999999", "+917777777777"]


# --- CallingHoursPolicy -------------------------------------------------


def _ist(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_calling_hours_inside_window_is_allowed() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    # 2026-05-06 is a Wednesday in IST
    assert p.can_call_now(_ist(2026, 5, 6, 14)) is True


def test_calling_hours_before_start_blocked() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    assert p.can_call_now(_ist(2026, 5, 6, 9)) is False


def test_calling_hours_at_end_excluded() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    assert p.can_call_now(_ist(2026, 5, 6, 19)) is False


def test_calling_hours_skips_sundays() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    # 2026-05-10 is a Sunday
    assert p.can_call_now(_ist(2026, 5, 10, 12)) is False


def test_calling_hours_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        CallingHoursPolicy(start="19:00", end="10:00")


def test_next_call_window_when_inside_returns_now() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    inside = _ist(2026, 5, 6, 14)
    assert p.next_call_window(inside) == inside


def test_next_call_window_when_before_start_returns_today_start() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    before = _ist(2026, 5, 6, 8)
    out = p.next_call_window(before)
    assert out == _ist(2026, 5, 6, 10)


def test_next_call_window_when_after_end_returns_tomorrow() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    after = _ist(2026, 5, 6, 22)
    out = p.next_call_window(after)
    assert out == _ist(2026, 5, 7, 10)


def test_next_call_window_skips_sunday() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    # Saturday evening, after-hours -> next legal slot is Monday 10:00
    sat_eve = _ist(2026, 5, 9, 22)
    out = p.next_call_window(sat_eve)
    assert out == _ist(2026, 5, 11, 10)  # Monday


def test_next_call_window_naive_datetime_assumed_ist() -> None:
    p = CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)
    naive = datetime(2026, 5, 6, 14)  # naive — should be assumed IST
    assert p.next_call_window(naive).hour == 14
