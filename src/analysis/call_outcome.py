"""Post-call lead outcome analysis.

Transport-agnostic: given a transcript (+ slots, optional telephony status),
returns a CallAnalysis. Conversational outcomes come from a single LLM JSON
pass; unreachable outcomes short-circuit from the telephony status (Task 4).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.campaign.models import (
    CONVERSATIONAL_OUTCOMES,
    CallAnalysis,
    LeadCallOutcome,
    outcome_from_telephony,
)
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage

log = logging.getLogger(__name__)

ANALYSIS_TIMEOUT_S = 15.0

_ACTION_FALLBACK = {
    "close_positive": LeadCallOutcome.INTERESTED,
    "schedule_callback": LeadCallOutcome.CALLBACK_REQUESTED,
    "transfer": LeadCallOutcome.ESCALATED,
    "close_negative": LeadCallOutcome.NOT_INTERESTED,
}

_SYSTEM_PROMPT = """You analyze a finished sales/outreach phone call and classify its outcome.

Return ONLY a JSON object with these keys:
- "outcome": one of ["interested", "callback_requested", "not_interested", "refused", "escalated", "angry_hostile"]
- "summary": 2-3 sentence recap of the call, in ENGLISH
- "notes": actionable observations (objections, preferences, next steps), in ENGLISH
- "callback_datetime": ISO-8601 datetime string in the given timezone if the lead asked to be called back at a resolvable time, else null
- "callback_phrase": the lead's raw words about timing (e.g. "kal shaam 5 baje"), else null

Outcome guidance:
- interested: lead is positively engaged / wants to proceed.
- callback_requested: lead asked to be contacted again later.
- not_interested: politely declines.
- refused: refuses to engage / wants to be left alone / do-not-call.
- escalated: needs/requests a human agent or transfer.
- angry_hostile: abusive, threatening, or very hostile.

Resolve relative times ("kal", "Monday", "do din baad") against NOW in the given TIMEZONE.
Write summary and notes in English even though the call is in Hindi/Hinglish."""


def _render_transcript(transcript: list[LLMMessage]) -> str:
    lines = []
    for m in transcript:
        if m.role == "system":
            continue
        who = "Agent" if m.role == "assistant" else "Lead"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def _resolve_callback(value: Any, tz: ZoneInfo) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("unknown tenant timezone %r; defaulting to Asia/Kolkata", tz_name)
        return ZoneInfo("Asia/Kolkata")


def _fallback(final_action: Optional[str], note: str) -> CallAnalysis:
    outcome = _ACTION_FALLBACK.get(final_action or "", LeadCallOutcome.NOT_INTERESTED)
    # summary is intentionally empty on fallback; the reason lives in notes
    return CallAnalysis(
        outcome=outcome,
        summary="",
        notes=f"(auto-derived; {note})",
        analysis_source="fallback",
    )


async def analyze_call(
    *,
    transcript: list[LLMMessage],
    slots: dict[str, Any],
    telephony_status: Optional[str],
    final_action: Optional[str],
    tenant_timezone: str,
    now: datetime,
    llm: ILLMProvider,
) -> CallAnalysis:
    """Classify a finished call. Never raises — failures fall back."""
    tz = _zone(tenant_timezone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    now_local = now.astimezone(tz)

    user_msg = (
        f"NOW: {now_local.isoformat()}\nTIMEZONE: {tenant_timezone}\n"
        f"COLLECTED DATA: {json.dumps(slots, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n{_render_transcript(transcript)}"
    )
    messages = [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]
    cfg = LLMConfig(temperature=0.2, max_tokens=512, response_format="json")

    try:
        result = await asyncio.wait_for(
            llm.generate(messages, cfg), timeout=ANALYSIS_TIMEOUT_S
        )
        obj = json.loads(result.text)
        outcome = LeadCallOutcome(obj["outcome"])
        if outcome not in CONVERSATIONAL_OUTCOMES:
            raise ValueError(f"non-conversational outcome from LLM: {outcome}")
    except Exception as exc:  # noqa: BLE001 - analysis must never crash the caller
        log.warning("call analysis LLM pass failed: %s", exc)
        return _fallback(final_action, f"analysis failed: {type(exc).__name__}")

    return CallAnalysis(
        outcome=outcome,
        summary=str(obj.get("summary") or ""),
        notes=str(obj.get("notes") or ""),
        callback_datetime=_resolve_callback(obj.get("callback_datetime"), tz),
        callback_phrase=(obj.get("callback_phrase") or None),
        analysis_source="llm",
    )
