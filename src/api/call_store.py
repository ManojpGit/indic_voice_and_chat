"""Persistence helpers for the call record (``conversations`` table).

Call Lead inserts an ``in_progress`` row keyed by the provider Call SID; at
teardown the bridge looks the row up by that SID and writes the outcome +
duration + cost. Cost is Σ(provider cost/min from the ``provider_costs`` catalog
for the providers actually used) × duration. Kept here so the endpoint and the
bridge teardown share one implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.conversation import Conversation
from src.models.tenant import ProviderCost

log = logging.getLogger(__name__)

# A call counts against the concurrency cap while it is being placed or is live.
ACTIVE_STATUSES = ("in_progress", "answered")


# --- Outcome persister hook ---------------------------------------------
# The bridges have no DB session of their own; at teardown they hand the
# (call_sid, outcome payload) to this hook, which the app wires in its lifespan
# to open a session and write the outcome + cost. Unset (tests / dev console
# without DB) → teardown is a clean no-op.
_persister: Optional[Callable[[str, dict], Awaitable[None]]] = None


def set_call_outcome_persister(fn: Optional[Callable[[str, dict], Awaitable[None]]]) -> None:
    global _persister
    _persister = fn


async def deliver_to_persister(call_sid: Optional[str], payload: dict) -> None:
    """Hand a finished call's outcome to the persister, if one is wired.

    Never raises — outcome persistence must not break call teardown.
    """
    if _persister is None or not call_sid:
        return
    try:
        await _persister(call_sid, payload)
    except Exception:  # noqa: BLE001 — teardown must survive a DB hiccup
        log.exception("call outcome persistence failed", extra={"sid": call_sid})


async def count_active_calls(session: AsyncSession, tenant_id: str) -> int:
    """How many of this tenant's calls are currently placing/live."""
    return (await session.execute(
        select(func.count()).select_from(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.status.in_(ACTIVE_STATUSES),
        )
    )).scalar_one()


def _providers_used(
    *, mode: Optional[str], stt_provider: Optional[str], llm_provider: Optional[str],
    tts_provider: Optional[str], realtime_provider: Optional[str],
    telephony_provider: Optional[str],
) -> list[tuple[str, str]]:
    """The (kind, provider) pairs billed for one call, by mode."""
    pairs: list[tuple[str, str]] = []
    if mode == "s2s":
        if realtime_provider:
            pairs.append(("s2s", realtime_provider))
    else:
        if stt_provider:
            pairs.append(("stt", stt_provider))
        if llm_provider:
            pairs.append(("llm", llm_provider))
        if tts_provider:
            pairs.append(("tts", tts_provider))
    if telephony_provider:
        pairs.append(("telephony", telephony_provider))
    return pairs


async def compute_call_cost(
    session: AsyncSession,
    *,
    mode: Optional[str],
    stt_provider: Optional[str] = None,
    llm_provider: Optional[str] = None,
    tts_provider: Optional[str] = None,
    realtime_provider: Optional[str] = None,
    telephony_provider: Optional[str] = None,
    duration_ms: Optional[int],
) -> float:
    """Σ(cost/min for the providers used) × duration. 0.0 if no duration."""
    if not duration_ms or duration_ms <= 0:
        return 0.0
    pairs = _providers_used(
        mode=mode, stt_provider=stt_provider, llm_provider=llm_provider,
        tts_provider=tts_provider, realtime_provider=realtime_provider,
        telephony_provider=telephony_provider,
    )
    minutes = duration_ms / 60_000.0
    total = 0.0
    for kind, provider in pairs:
        row = await session.get(ProviderCost, (kind, provider))
        if row is not None:
            total += row.cost_per_min * minutes
    return round(total, 6)


async def record_outcome(
    session: AsyncSession,
    provider_call_sid: str,
    *,
    status: str = "ended",
    outcome: Optional[str] = None,
    summary: Optional[str] = None,
    notes: Optional[str] = None,
    callback_at: Optional[datetime] = None,
    duration_ms: Optional[int] = None,
    ended_at: Optional[datetime] = None,
) -> Optional[Conversation]:
    """Find the call row by provider Call SID and write its outcome + cost.

    Cost is computed from the providers recorded on the row + ``duration_ms``.
    Returns the updated row, or None if no row matches the SID.
    """
    row = (await session.execute(
        select(Conversation).where(Conversation.provider_call_sid == provider_call_sid)
    )).scalar_one_or_none()
    if row is None:
        log.warning("no conversation for call sid", extra={"sid": provider_call_sid})
        return None

    row.status = status
    if outcome is not None:
        row.outcome = outcome
    if summary is not None:
        row.summary = summary
    if notes is not None:
        row.notes = notes
    if callback_at is not None:
        row.callback_at = callback_at
    row.ended_at = ended_at or datetime.utcnow()
    if duration_ms is not None:
        row.duration_ms = duration_ms
    elif row.duration_ms is None and row.started_at is not None:
        # Derive call duration from when the row was created (call placed).
        row.duration_ms = max(0, int((row.ended_at - row.started_at).total_seconds() * 1000))
    row.cost = await compute_call_cost(
        session,
        mode=row.mode,
        stt_provider=row.stt_provider,
        llm_provider=row.llm_provider,
        tts_provider=row.tts_provider,
        realtime_provider=row.realtime_provider,
        telephony_provider=row.telephony_provider,
        duration_ms=row.duration_ms,
    )
    await session.commit()
    return row
