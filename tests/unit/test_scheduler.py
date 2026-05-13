from __future__ import annotations

from datetime import datetime, timedelta

from src.campaign.dnd_filter import IST, CallingHoursPolicy, DNDFilter, InMemoryDNDStore
from src.campaign.models import Lead, LeadStatus
from src.campaign.scheduler import (
    CallScheduler,
    RateLimitConfig,
    RetryConfig,
)


def _ist(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def _hours() -> CallingHoursPolicy:
    return CallingHoursPolicy(start="10:00", end="19:00", skip_weekday=6)


def _dnd(*nums: str) -> DNDFilter:
    return DNDFilter(InMemoryDNDStore(list(nums)))


def _lead(id: str, phone: str = "+919999999999", **kw) -> Lead:
    return Lead(id=id, tenant_id="t_test", phone_number=phone, **kw)


# --- can-call gating ----------------------------------------------------


def test_poll_blocks_outside_hours() -> None:
    sched = CallScheduler(hours=_hours(), dnd_filter=_dnd())
    out = sched.poll(leads=[_lead("l1")], active_count=0, now=_ist(2026, 5, 6, 9))
    assert out.blocked_by_hours is True
    assert out.next_eligible_at == _ist(2026, 5, 6, 10)
    assert out.leads == []


def test_poll_blocks_concurrency_cap() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        rate_limit=RateLimitConfig(max_concurrent_calls=3, calls_per_minute=100),
    )
    out = sched.poll(
        leads=[_lead("l1")],
        active_count=3,
        now=_ist(2026, 5, 6, 14),
    )
    assert out.blocked_by_concurrency is True
    assert out.leads == []


def test_poll_blocks_rate_limit_after_burst() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        rate_limit=RateLimitConfig(calls_per_minute=2, max_concurrent_calls=10),
    )
    now = _ist(2026, 5, 6, 14)
    sched.mark_attempted(now)
    sched.mark_attempted(now)
    out = sched.poll(leads=[_lead("l1")], active_count=0, now=now)
    assert out.blocked_by_rate is True
    assert out.next_eligible_at == now + timedelta(seconds=60)


def test_poll_picks_leads_within_budget() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        rate_limit=RateLimitConfig(calls_per_minute=10, max_concurrent_calls=5),
    )
    leads = [_lead(f"l{i}") for i in range(20)]
    out = sched.poll(leads=leads, active_count=0, now=_ist(2026, 5, 6, 14))
    # Limited by max_concurrent (5) since active_count=0
    assert len(out.leads) == 5


def test_poll_max_pick_param() -> None:
    sched = CallScheduler(hours=_hours(), dnd_filter=_dnd())
    leads = [_lead(f"l{i}") for i in range(20)]
    out = sched.poll(leads=leads, active_count=0, now=_ist(2026, 5, 6, 14), max_pick=2)
    assert len(out.leads) == 2


# --- per-lead eligibility -----------------------------------------------


def test_poll_skips_dnd_listed_leads() -> None:
    sched = CallScheduler(hours=_hours(), dnd_filter=_dnd("+919999999999"))
    leads = [
        _lead("l1", phone="+919999999999"),  # blocked
        _lead("l2", phone="+918888888888"),  # allowed
    ]
    out = sched.poll(leads=leads, active_count=0, now=_ist(2026, 5, 6, 14))
    assert [lead.id for lead in out.leads] == ["l2"]


def test_poll_skips_in_flight_or_completed_leads() -> None:
    sched = CallScheduler(hours=_hours(), dnd_filter=_dnd())
    leads = [
        _lead("l1", status=LeadStatus.IN_FLIGHT),
        _lead("l2", status=LeadStatus.COMPLETED),
        _lead("l3", status=LeadStatus.PENDING),
    ]
    out = sched.poll(leads=leads, active_count=0, now=_ist(2026, 5, 6, 14))
    assert [lead.id for lead in out.leads] == ["l3"]


def test_poll_respects_next_retry_at() -> None:
    sched = CallScheduler(hours=_hours(), dnd_filter=_dnd())
    later = _ist(2026, 5, 6, 16)
    leads = [
        _lead("l1", status=LeadStatus.RETRY, next_retry_at=later),
        _lead("l2", status=LeadStatus.RETRY, next_retry_at=_ist(2026, 5, 6, 12)),
    ]
    # Now is 14:00 — l1 (16:00 retry) is too early, l2 (12:00 retry) is ready
    out = sched.poll(leads=leads, active_count=0, now=_ist(2026, 5, 6, 14))
    assert [lead.id for lead in out.leads] == ["l2"]


# --- retry scheduling ---------------------------------------------------


def test_schedule_retry_first_attempt_sets_retry_status() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        retry=RetryConfig(max_retry_attempts=3, retry_interval_hours=2),
    )
    lead = _lead("l1")
    now = _ist(2026, 5, 6, 14)
    sched.schedule_retry(lead, now=now)
    assert lead.status is LeadStatus.RETRY
    assert lead.retry_count == 1
    assert lead.next_retry_at == now + timedelta(hours=2)


def test_schedule_retry_after_max_attempts_marks_failed() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        retry=RetryConfig(max_retry_attempts=3, retry_interval_hours=2),
    )
    lead = _lead("l1", retry_count=2)
    sched.schedule_retry(lead, now=_ist(2026, 5, 6, 14))
    assert lead.status is LeadStatus.FAILED
    assert lead.retry_count == 3
    assert lead.next_retry_at is None


# --- rate-limit window evicts stale entries -----------------------------


def test_rate_limit_window_evicts_old_entries() -> None:
    sched = CallScheduler(
        hours=_hours(),
        dnd_filter=_dnd(),
        rate_limit=RateLimitConfig(calls_per_minute=2, max_concurrent_calls=10),
    )
    t0 = _ist(2026, 5, 6, 14, 0)
    sched.mark_attempted(t0)
    sched.mark_attempted(t0)
    # Now move past the 60s window — old entries should evict.
    later = t0 + timedelta(seconds=70)
    out = sched.poll(leads=[_lead("l1")], active_count=0, now=later)
    assert out.blocked_by_rate is False
    assert len(out.leads) == 1
