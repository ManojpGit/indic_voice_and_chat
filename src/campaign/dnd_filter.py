"""DND (Do Not Disturb) compliance filter + calling-hours window check.

The DND list is intentionally simple — an in-memory set keyed by normalized
phone number — so tests don't need a database. Production will plug a
PostgreSQL-backed implementation behind ``IDNDStore`` (TODO Phase 6+).

Calling-hours behavior:
- Configured per campaign in IST (Asia/Kolkata).
- ``can_call_now`` returns True iff ``now`` (default: ``datetime.now(IST)``)
  is inside ``[start, end]`` and not on a Sunday.
- ``next_call_window`` returns when the next legal call window opens, useful
  for scheduling deferred retries.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Optional, Protocol


# IST is fixed at +05:30 — no DST transitions, so a static timezone is fine.
IST = timezone(timedelta(hours=5, minutes=30))


# --- Phone normalization ------------------------------------------------


_PHONE_NORMALIZE = re.compile(r"[^0-9+]")


def normalize_phone(number: str) -> str:
    """Strip whitespace + punctuation. Preserves leading ``+``.

    "+91 (999) 999-9999" -> "+919999999999"
    """
    if not number:
        return ""
    cleaned = _PHONE_NORMALIZE.sub("", number)
    return cleaned


# --- DND store interface ------------------------------------------------


class IDNDStore(Protocol):
    def add(self, number: str) -> None: ...

    def remove(self, number: str) -> None: ...

    def is_dnd(self, number: str) -> bool: ...

    def __len__(self) -> int: ...


class InMemoryDNDStore:
    """Process-local DND list. Good enough for Phase 5 + tests."""

    def __init__(self, initial: Optional[Iterable[str]] = None) -> None:
        self._set: set[str] = set()
        if initial:
            for n in initial:
                self.add(n)

    def add(self, number: str) -> None:
        n = normalize_phone(number)
        if n:
            self._set.add(n)

    def remove(self, number: str) -> None:
        n = normalize_phone(number)
        self._set.discard(n)

    def is_dnd(self, number: str) -> bool:
        return normalize_phone(number) in self._set

    def __len__(self) -> int:
        return len(self._set)


# --- DND filter ---------------------------------------------------------


class DNDFilter:
    def __init__(self, store: IDNDStore, enabled: bool = True) -> None:
        self._store = store
        self._enabled = enabled

    @property
    def store(self) -> IDNDStore:
        return self._store

    def is_blocked(self, number: str) -> bool:
        if not self._enabled:
            return False
        return self._store.is_dnd(number)

    def filter_blocked(self, numbers: Iterable[str]) -> tuple[list[str], list[str]]:
        """Returns ``(allowed, blocked)`` lists, preserving input order."""
        allowed: list[str] = []
        blocked: list[str] = []
        for n in numbers:
            if self.is_blocked(n):
                blocked.append(n)
            else:
                allowed.append(n)
        return allowed, blocked


# --- Calling hours ------------------------------------------------------


class CallingHoursPolicy:
    """Calling hours window. PRD §5.1 default: 10:00–19:00 IST, Mon–Sat."""

    def __init__(
        self,
        start: str = "10:00",
        end: str = "19:00",
        skip_weekday: Optional[int] = 6,  # 6 = Sunday
        tz: timezone = IST,
    ) -> None:
        self._start = _parse_hhmm(start)
        self._end = _parse_hhmm(end)
        if self._end <= self._start:
            raise ValueError("calling-hours end must be after start")
        self._skip_weekday = skip_weekday
        self._tz = tz

    @property
    def tz(self) -> timezone:
        return self._tz

    def can_call_now(self, now: Optional[datetime] = None) -> bool:
        when = self._coerce(now)
        if self._skip_weekday is not None and when.weekday() == self._skip_weekday:
            return False
        return self._start <= when.time() < self._end

    def next_call_window(self, after: Optional[datetime] = None) -> datetime:
        """Return the next ``datetime`` (in policy tz) when calling is legal."""
        cursor = self._coerce(after)
        for _ in range(8):  # max 7 days lookahead is plenty
            window_start = cursor.replace(
                hour=self._start.hour,
                minute=self._start.minute,
                second=0,
                microsecond=0,
            )
            window_end = cursor.replace(
                hour=self._end.hour,
                minute=self._end.minute,
                second=0,
                microsecond=0,
            )
            is_skip_day = (
                self._skip_weekday is not None
                and cursor.weekday() == self._skip_weekday
            )
            if not is_skip_day:
                if cursor < window_start:
                    return window_start
                if cursor < window_end:
                    return cursor
            # Move to next day's window start.
            cursor = (cursor + timedelta(days=1)).replace(
                hour=self._start.hour,
                minute=self._start.minute,
                second=0,
                microsecond=0,
            )
        # Fallback — shouldn't happen with sensible config.
        return cursor

    def _coerce(self, when: Optional[datetime]) -> datetime:
        if when is None:
            return datetime.now(self._tz)
        if when.tzinfo is None:
            return when.replace(tzinfo=self._tz)
        return when.astimezone(self._tz)


def _parse_hhmm(s: str) -> time:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"expected HH:MM, got {s!r}")
    hh, mm = int(parts[0]), int(parts[1])
    return time(hh, mm)
