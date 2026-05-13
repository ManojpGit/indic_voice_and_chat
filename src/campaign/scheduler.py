"""Call scheduler + retry logic.

Owns three policies:
1. Calling-hours window (delegated to ``CallingHoursPolicy``).
2. Outbound rate limit (calls/minute) implemented as a sliding window.
3. Retry timing (max attempts + interval, schedules ``next_retry_at``).

The scheduler is purely advisory — it picks which lead to call next from
an in-memory queue and tells the orchestrator when it's allowed to dial.
The orchestrator owns the actual dispatch and concurrency cap.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional

from src.campaign.dnd_filter import IST, CallingHoursPolicy, DNDFilter
from src.campaign.models import Lead, LeadStatus


@dataclass
class RetryConfig:
    max_retry_attempts: int = 3
    retry_interval_hours: int = 2


@dataclass
class RateLimitConfig:
    calls_per_minute: int = 20
    max_concurrent_calls: int = 10


@dataclass
class SchedulerDecision:
    """What ``poll`` returns."""

    leads: list[Lead] = field(default_factory=list)
    blocked_by_hours: bool = False
    blocked_by_rate: bool = False
    blocked_by_concurrency: bool = False
    next_eligible_at: Optional[datetime] = None


class CallScheduler:
    def __init__(
        self,
        hours: CallingHoursPolicy,
        dnd_filter: DNDFilter,
        retry: Optional[RetryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ) -> None:
        self._hours = hours
        self._dnd = dnd_filter
        self._retry = retry or RetryConfig()
        self._rate = rate_limit or RateLimitConfig()
        self._dispatched_at: collections.deque[datetime] = collections.deque()

    @property
    def retry_config(self) -> RetryConfig:
        return self._retry

    # --- Lead-state transitions -----------------------------------------

    def mark_attempted(self, when: Optional[datetime] = None) -> None:
        """Record a dispatched call for rate-limit accounting."""
        ts = self._now(when)
        self._dispatched_at.append(ts)
        self._evict_stale(ts)

    def schedule_retry(self, lead: Lead, now: Optional[datetime] = None) -> Lead:
        """Bump retry counter, set ``next_retry_at``, mark RETRY or FAILED."""
        when = self._now(now)
        lead.retry_count += 1
        if lead.retry_count >= self._retry.max_retry_attempts:
            lead.status = LeadStatus.FAILED
            lead.next_retry_at = None
            return lead
        lead.status = LeadStatus.RETRY
        lead.next_retry_at = when + timedelta(hours=self._retry.retry_interval_hours)
        return lead

    # --- Polling --------------------------------------------------------

    def poll(
        self,
        leads: Iterable[Lead],
        active_count: int,
        now: Optional[datetime] = None,
        max_pick: Optional[int] = None,
    ) -> SchedulerDecision:
        """Return up to ``max_pick`` leads ready to dial right now."""
        when = self._now(now)
        decision = SchedulerDecision()

        if not self._hours.can_call_now(when):
            decision.blocked_by_hours = True
            decision.next_eligible_at = self._hours.next_call_window(when)
            return decision

        if active_count >= self._rate.max_concurrent_calls:
            decision.blocked_by_concurrency = True
            return decision

        self._evict_stale(when)
        rate_room = max(0, self._rate.calls_per_minute - len(self._dispatched_at))
        if rate_room == 0:
            decision.blocked_by_rate = True
            # Earliest the rate window will free a slot
            if self._dispatched_at:
                decision.next_eligible_at = self._dispatched_at[0] + timedelta(seconds=60)
            return decision

        concurrency_room = self._rate.max_concurrent_calls - active_count
        budget = min(rate_room, concurrency_room)
        if max_pick is not None:
            budget = min(budget, max_pick)

        for lead in leads:
            if budget <= 0:
                break
            if not self._lead_eligible(lead, when):
                continue
            decision.leads.append(lead)
            budget -= 1
        return decision

    # --- Internals ------------------------------------------------------

    def _lead_eligible(self, lead: Lead, when: datetime) -> bool:
        if lead.status not in (LeadStatus.PENDING, LeadStatus.RETRY):
            return False
        if self._dnd.is_blocked(lead.phone_number):
            return False
        if lead.next_retry_at is not None:
            target = lead.next_retry_at
            if target.tzinfo is None:
                target = target.replace(tzinfo=IST)
            if when < target:
                return False
        return True

    def _evict_stale(self, when: Optional[datetime] = None) -> None:
        threshold = self._now(when) - timedelta(seconds=60)
        while self._dispatched_at and self._dispatched_at[0] < threshold:
            self._dispatched_at.popleft()

    def _now(self, when: Optional[datetime]) -> datetime:
        if when is None:
            return datetime.now(IST)
        if when.tzinfo is None:
            return when.replace(tzinfo=IST)
        return when.astimezone(IST)
