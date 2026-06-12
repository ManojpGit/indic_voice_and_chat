"""Shared call-outcome recording for telephony bridges.

Telephony has no live UI (unlike the browser dev console), so on call-end we
analyze the finished call and *log* the outcome — the hook for DB persistence
later. Mixed into the Twilio/Exotel bridges so the logic lives in one place.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.analysis.call_outcome import analyze_agent_call
from src.interfaces.llm import ILLMProvider

log = logging.getLogger(__name__)


class OutcomeRecorderMixin:
    """Adds ``_record_outcome()`` to a bridge.

    The host bridge must set (in its ``__init__``): ``self._agent``,
    ``self._llm``, ``self._tenant_timezone``, ``self._last_action``, and
    ``self._outcome_recorded = False``.
    """

    # Declared for type-checkers; real values are set by the host's __init__.
    _agent: object
    _llm: Optional[ILLMProvider]
    _tenant_timezone: str
    _last_action: Optional[str]
    _outcome_recorded: bool

    async def _record_outcome(self) -> None:
        """Analyze the finished call and log its outcome. Idempotent; no-op
        without an LLM. Never raises — analysis must not break teardown."""
        if self._outcome_recorded or self._llm is None:
            return
        self._outcome_recorded = True
        try:
            analysis = await analyze_agent_call(
                self._agent,
                llm=self._llm,
                tenant_timezone=self._tenant_timezone,
                final_action=self._last_action,
                now=datetime.now(timezone.utc),
            )
        except Exception:  # noqa: BLE001 - never let analysis break teardown
            log.exception("call outcome analysis failed")
            return
        if analysis is None:
            return
        cb = analysis.callback_datetime
        log.info(
            "call outcome",
            extra={
                "outcome": analysis.outcome.value,
                "source": analysis.analysis_source,
                "summary": analysis.summary[:200],
                "callback": cb.isoformat() if cb else None,
            },
        )
        # Persist to the conversations row (keyed by the provider Call SID), if
        # the host knows its SID and a persister is wired. No-op otherwise.
        from src.api import call_store
        call_sid = getattr(self, "_provider_call_sid", None) or getattr(self, "_call_sid", None)
        await call_store.deliver_to_persister(call_sid, {
            "type": "outcome", "outcome": analysis.outcome.value,
            "summary": analysis.summary, "notes": analysis.notes,
            "callback_datetime": cb.isoformat() if cb else None,
            "source": analysis.analysis_source,
        })
