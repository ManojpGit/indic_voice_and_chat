# Post-call Lead Outcome Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify a finished call into a 10-value `LeadCallOutcome`, with an English summary + notes and a tenant-timezone-resolved callback datetime, surfaced in the dev console and fed into the campaign pipeline.

**Architecture:** A transport-agnostic analyzer (`src/analysis/call_outcome.py`) takes a transcript + slots + optional telephony status and returns a `CallAnalysis`. Unreachable outcomes short-circuit from the telephony status; conversational outcomes come from one post-call LLM JSON pass. A mapping keeps the legacy `CallDisposition` consumers (orchestrator/CRM/benchmarks) working.

**Tech Stack:** Python 3.12, Pydantic, `zoneinfo` (stdlib), pytest + pytest-asyncio. Reuses the tenant's configured LLM (Gemini) via `ILLMProvider`.

Spec: `docs/superpowers/specs/2026-06-05-call-outcome-analysis-design.md`.

---

## File Structure

- **Create** `src/analysis/__init__.py` — new package marker.
- **Create** `src/analysis/call_outcome.py` — `analyze_call()`, prompt builder, callback-datetime resolution, fallback.
- **Modify** `src/campaign/models.py` — add `LeadCallOutcome` enum, `CallAnalysis` model, `outcome_from_telephony()`, `disposition_from_outcome()`; extend `CallResult`.
- **Modify** `src/config_tenant.py` — add `timezone` field to `TenantSettings`.
- **Modify** `src/api/browser_bridge.py` — track last action; analyze on call end; emit `outcome` WS message.
- **Modify** `static/dev_console.html` — outcome results panel.
- **Modify** `src/models/conversation.py` — add `outcome`, `summary`, `notes`, `callback_at` columns.
- **Create** `tests/unit/test_call_outcome_mappings.py`, `tests/unit/test_call_outcome_analyzer.py` — and extend `tests/unit/test_browser_bridge.py`, `tests/unit/test_campaign_models.py`.

---

## Task 1: Outcome enum + mapping tables

**Files:**
- Modify: `src/campaign/models.py`
- Test: `tests/unit/test_call_outcome_mappings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_call_outcome_mappings.py
import pytest

from src.campaign.models import (
    CallDisposition,
    LeadCallOutcome,
    disposition_from_outcome,
    outcome_from_telephony,
)


def test_every_outcome_maps_to_a_disposition():
    for outcome in LeadCallOutcome:
        assert isinstance(disposition_from_outcome(outcome), CallDisposition)


def test_dnd_outcomes():
    assert disposition_from_outcome(LeadCallOutcome.REFUSED) == CallDisposition.DND_REQUESTED
    assert disposition_from_outcome(LeadCallOutcome.ANGRY_HOSTILE) == CallDisposition.DND_REQUESTED


def test_qualifying_outcomes():
    assert disposition_from_outcome(LeadCallOutcome.INTERESTED) == CallDisposition.INTERESTED_TRANSFER
    assert disposition_from_outcome(LeadCallOutcome.CALLBACK_REQUESTED) == CallDisposition.INTERESTED_CALLBACK
    assert disposition_from_outcome(LeadCallOutcome.ESCALATED) == CallDisposition.INTERESTED_TRANSFER


def test_retryable_outcomes():
    for o in (LeadCallOutcome.NO_ANSWER, LeadCallOutcome.BUSY, LeadCallOutcome.CALL_FAILED):
        assert disposition_from_outcome(o) == CallDisposition.BUSY_RETRY
    assert disposition_from_outcome(LeadCallOutcome.VOICEMAIL) == CallDisposition.VOICEMAIL


def test_telephony_status_maps_to_outcome():
    assert outcome_from_telephony("no_answer") == LeadCallOutcome.NO_ANSWER
    assert outcome_from_telephony("busy") == LeadCallOutcome.BUSY
    assert outcome_from_telephony("failed") == LeadCallOutcome.CALL_FAILED
    assert outcome_from_telephony("voicemail") == LeadCallOutcome.VOICEMAIL


def test_telephony_status_unknown_returns_none():
    assert outcome_from_telephony("answered") is None
    assert outcome_from_telephony(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_mappings.py -v`
Expected: FAIL — `ImportError: cannot import name 'LeadCallOutcome'`.

- [ ] **Step 3: Add the enum and mappings**

In `src/campaign/models.py`, directly after the existing `class CallDisposition(str, Enum)` block, add:

```python
class LeadCallOutcome(str, Enum):
    INTERESTED = "interested"
    CALLBACK_REQUESTED = "callback_requested"
    NOT_INTERESTED = "not_interested"
    REFUSED = "refused"
    ESCALATED = "escalated"
    ANGRY_HOSTILE = "angry_hostile"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"
    BUSY = "busy"
    CALL_FAILED = "call_failed"


# Conversational outcomes the LLM may return (the rest come from telephony).
CONVERSATIONAL_OUTCOMES = frozenset({
    LeadCallOutcome.INTERESTED,
    LeadCallOutcome.CALLBACK_REQUESTED,
    LeadCallOutcome.NOT_INTERESTED,
    LeadCallOutcome.REFUSED,
    LeadCallOutcome.ESCALATED,
    LeadCallOutcome.ANGRY_HOSTILE,
})

_OUTCOME_TO_DISPOSITION = {
    LeadCallOutcome.INTERESTED: CallDisposition.INTERESTED_TRANSFER,
    LeadCallOutcome.CALLBACK_REQUESTED: CallDisposition.INTERESTED_CALLBACK,
    LeadCallOutcome.NOT_INTERESTED: CallDisposition.NOT_INTERESTED,
    LeadCallOutcome.REFUSED: CallDisposition.DND_REQUESTED,
    LeadCallOutcome.ESCALATED: CallDisposition.INTERESTED_TRANSFER,
    LeadCallOutcome.ANGRY_HOSTILE: CallDisposition.DND_REQUESTED,
    LeadCallOutcome.NO_ANSWER: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.BUSY: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.CALL_FAILED: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.VOICEMAIL: CallDisposition.VOICEMAIL,
}

_TELEPHONY_TO_OUTCOME = {
    "no_answer": LeadCallOutcome.NO_ANSWER,
    "busy": LeadCallOutcome.BUSY,
    "failed": LeadCallOutcome.CALL_FAILED,
    "voicemail": LeadCallOutcome.VOICEMAIL,
}


def disposition_from_outcome(outcome: "LeadCallOutcome") -> CallDisposition:
    """Map a canonical outcome to the legacy disposition consumed by the
    orchestrator/CRM/benchmarks. Total over LeadCallOutcome."""
    return _OUTCOME_TO_DISPOSITION[outcome]


def outcome_from_telephony(status: Optional[str]) -> Optional["LeadCallOutcome"]:
    """Map a normalized telephony status to an unreachable outcome, or None
    when the call connected (the conversational path then applies)."""
    return _TELEPHONY_TO_OUTCOME.get(status or "")
```

`Optional` is already imported in `src/campaign/models.py` (used by existing models).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_mappings.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/campaign/models.py tests/unit/test_call_outcome_mappings.py
git commit -m "feat(outcome): LeadCallOutcome enum + telephony/disposition mappings"
```

---

## Task 2: `CallAnalysis` model + tenant timezone

**Files:**
- Modify: `src/campaign/models.py`
- Modify: `src/config_tenant.py:158-165` (`TenantSettings`)
- Test: `tests/unit/test_call_outcome_mappings.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_call_outcome_mappings.py`:

```python
def test_call_analysis_defaults():
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    a = CallAnalysis(outcome=LeadCallOutcome.INTERESTED)
    assert a.summary == ""
    assert a.notes == ""
    assert a.callback_datetime is None
    assert a.callback_phrase is None
    assert a.analysis_source == "llm"


def test_tenant_settings_has_timezone_default():
    from src.config_tenant import TenantSettings

    t = TenantSettings(id="t_x", slug="x", name="X")
    assert t.timezone == "Asia/Kolkata"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_mappings.py::test_call_analysis_defaults tests/unit/test_call_outcome_mappings.py::test_tenant_settings_has_timezone_default -v`
Expected: FAIL — `ImportError: cannot import name 'CallAnalysis'`.

- [ ] **Step 3: Add `CallAnalysis` model**

In `src/campaign/models.py`, after the `LeadCallOutcome` block from Task 1, add (the file already imports `datetime`, `Optional`, `Any`, `BaseModel`, `Field`):

```python
class CallAnalysis(BaseModel):
    """Result of analyzing one finished call. Produced by analyze_call()."""

    outcome: LeadCallOutcome
    summary: str = ""           # English, 2-3 sentences
    notes: str = ""             # English; objections, preferences, next steps
    callback_datetime: Optional[datetime] = None  # tz-aware when resolved
    callback_phrase: Optional[str] = None         # raw, e.g. "kal shaam 5 baje"
    analysis_source: str = "llm"  # "llm" | "telephony" | "fallback"
```

- [ ] **Step 4: Add the `timezone` field**

In `src/config_tenant.py`, inside `class TenantSettings(BaseModel)` immediately after the `default_language: str = "hi"` line (line ~165), add:

```python
    timezone: str = "Asia/Kolkata"  # IANA tz; resolves relative callback times
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_mappings.py -v`
Expected: PASS (8 passed).

- [ ] **Step 6: Commit**

```bash
git add src/campaign/models.py src/config_tenant.py tests/unit/test_call_outcome_mappings.py
git commit -m "feat(outcome): CallAnalysis model + tenant timezone field"
```

---

## Task 3: Analyzer — conversational LLM path

**Files:**
- Create: `src/analysis/__init__.py`
- Create: `src/analysis/call_outcome.py`
- Test: `tests/unit/test_call_outcome_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_call_outcome_analyzer.py
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.analysis.call_outcome import analyze_call
from src.campaign.models import LeadCallOutcome
from src.interfaces.llm import LLMMessage, LLMResult


class FakeLLM:
    """Returns a canned JSON string and records the call."""

    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    async def generate(self, messages, config) -> LLMResult:
        self.calls += 1
        self.last_messages = messages
        return LLMResult(text=self._text, finish_reason="stop")

    async def generate_stream(self, messages, config):  # pragma: no cover
        raise NotImplementedError


TRANSCRIPT = [
    LLMMessage(role="assistant", content="Namaste Raju ji, ek minute hai?"),
    LLMMessage(role="user", content="Haan bataiye"),
]


@pytest.mark.asyncio
async def test_conversational_outcome_parsed():
    llm = FakeLLM(
        '{"outcome": "interested", "summary": "Lead was interested.", '
        '"notes": "Wants the app link.", "callback_datetime": null, '
        '"callback_phrase": null}'
    )
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="close_positive", tenant_timezone="Asia/Kolkata",
        now=now, llm=llm,
    )
    assert llm.calls == 1
    assert result.outcome == LeadCallOutcome.INTERESTED
    assert result.summary == "Lead was interested."
    assert result.analysis_source == "llm"


@pytest.mark.asyncio
async def test_callback_datetime_resolved_tz_aware():
    llm = FakeLLM(
        '{"outcome": "callback_requested", "summary": "Asked to call back.", '
        '"notes": "Busy now.", "callback_datetime": "2026-06-06T17:00:00", '
        '"callback_phrase": "kal shaam 5 baje"}'
    )
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="schedule_callback", tenant_timezone="Asia/Kolkata",
        now=now, llm=llm,
    )
    assert result.outcome == LeadCallOutcome.CALLBACK_REQUESTED
    assert result.callback_datetime == datetime(2026, 6, 6, 17, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert result.callback_phrase == "kal shaam 5 baje"


@pytest.mark.asyncio
async def test_vague_callback_is_null_with_phrase():
    llm = FakeLLM(
        '{"outcome": "callback_requested", "summary": "Call later.", '
        '"notes": "Unspecified time.", "callback_datetime": null, '
        '"callback_phrase": "baad mein"}'
    )
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="schedule_callback", tenant_timezone="Asia/Kolkata",
        now=now, llm=llm,
    )
    assert result.callback_datetime is None
    assert result.callback_phrase == "baad mein"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis'`.

- [ ] **Step 3: Create the package marker**

```python
# src/analysis/__init__.py
```
(empty file)

- [ ] **Step 4: Implement the analyzer (conversational path)**

```python
# src/analysis/call_outcome.py
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

    user_msg = (
        f"NOW: {now.isoformat()}\nTIMEZONE: {tenant_timezone}\n"
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_analyzer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/analysis/__init__.py src/analysis/call_outcome.py tests/unit/test_call_outcome_analyzer.py
git commit -m "feat(outcome): analyze_call conversational LLM path"
```

---

## Task 4: Analyzer — telephony short-circuit + fallback

**Files:**
- Modify: `src/analysis/call_outcome.py`
- Test: `tests/unit/test_call_outcome_analyzer.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_call_outcome_analyzer.py`:

```python
@pytest.mark.asyncio
async def test_telephony_status_short_circuits_without_llm():
    llm = FakeLLM("{}")
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=[], slots={}, telephony_status="busy",
        final_action=None, tenant_timezone="Asia/Kolkata", now=now, llm=llm,
    )
    assert result.outcome == LeadCallOutcome.BUSY
    assert result.analysis_source == "telephony"
    assert llm.calls == 0


class RaisingLLM:
    calls = 0

    async def generate(self, messages, config):
        RaisingLLM.calls += 1
        raise RuntimeError("boom")

    async def generate_stream(self, messages, config):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_action():
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="transfer", tenant_timezone="Asia/Kolkata",
        now=now, llm=RaisingLLM(),
    )
    assert result.outcome == LeadCallOutcome.ESCALATED
    assert result.analysis_source == "fallback"
    assert "auto-derived" in result.notes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_analyzer.py::test_telephony_status_short_circuits_without_llm -v`
Expected: FAIL — outcome is the LLM path / `llm.calls == 1`, not the telephony short-circuit.

- [ ] **Step 3: Add the short-circuit**

In `src/analysis/call_outcome.py`, inside `analyze_call`, immediately after `tz = _zone(tenant_timezone)` and before building `user_msg`, insert:

```python
    unreachable = outcome_from_telephony(telephony_status)
    if unreachable is not None:
        canned = {
            LeadCallOutcome.NO_ANSWER: "No answer.",
            LeadCallOutcome.BUSY: "Line was busy.",
            LeadCallOutcome.CALL_FAILED: "Call failed to connect.",
            LeadCallOutcome.VOICEMAIL: "Reached voicemail.",
        }
        return CallAnalysis(
            outcome=unreachable,
            summary=canned.get(unreachable, ""),
            analysis_source="telephony",
        )
```

(The `RaisingLLM` fallback already works via the existing `try/except` from Task 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_call_outcome_analyzer.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/call_outcome.py tests/unit/test_call_outcome_analyzer.py
git commit -m "feat(outcome): telephony short-circuit + fallback in analyze_call"
```

---

## Task 5: Dev console — analyze on call end, emit outcome

**Files:**
- Modify: `src/api/browser_bridge.py` (`__init__`, `_dispatch_utterance`, `_dispatch_text_turn`, `run` finally)
- Test: `tests/unit/test_browser_bridge.py` (append)

Note the bridge's `FakeAgent`/`FakeWebSocket` already exist in `tests/unit/test_browser_bridge.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_browser_bridge.py` (reuses the existing `FakeWebSocket`, `FakeAgent`, `_bridge` helpers):

```python
@pytest.mark.asyncio
async def test_emit_outcome_sends_ws_message(monkeypatch):
    import src.api.browser_bridge as bb
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    async def fake_analyze(**kwargs):
        return CallAnalysis(
            outcome=LeadCallOutcome.INTERESTED,
            summary="Good call.", notes="Send link.",
        )

    monkeypatch.setattr(bb, "analyze_call", fake_analyze)

    ws = FakeWebSocket([])
    bridge = _bridge(ws, FakeAgent())
    bridge._llm = object()  # non-None; fake_analyze ignores it
    await bridge._emit_outcome()

    sent = [json.loads(m) for m in ws.sent_text]
    outcome_msgs = [m for m in sent if m.get("type") == "outcome"]
    assert outcome_msgs and outcome_msgs[0]["outcome"] == "interested"
    assert outcome_msgs[0]["summary"] == "Good call."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge.py::test_emit_outcome_sends_ws_message -v`
Expected: FAIL — `AttributeError: 'BrowserVoiceBridge' object has no attribute '_emit_outcome'`.

- [ ] **Step 3: Add imports, constructor params, last-action tracking**

In `src/api/browser_bridge.py`, add near the other imports:

```python
from src.analysis.call_outcome import analyze_call
from src.interfaces.llm import LLMMessage
```

Add two optional params to `BrowserVoiceBridge.__init__` (so the LLM + timezone come from the factory, not by reaching through the agent). Change the signature to:

```python
    def __init__(
        self,
        websocket,
        agent,
        vad: VADDetector,
        config: BrowserBridgeConfig | None = None,
        stream_provider=None,
        llm=None,
        tenant_timezone: str = "Asia/Kolkata",
    ):
```

In `__init__`, after `self._cancel_event = None`, add:

```python
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._last_action: str | None = None
        self._outcome_emitted = False
```

In `_dispatch_utterance`, after `outcome = await self._agent.handle_turn(captured, self._send_pcm)`:

```python
        self._last_action = outcome.response.action
```

In `_dispatch_text_turn`, after the `finally:` block (where `self._cancel_event = None` is set), before the `if outcome.pipeline.cancelled:` check:

```python
        self._last_action = outcome.response.action
```

- [ ] **Step 4: Add `_emit_outcome`, call it on end, wire the factory**

Add this method to `BrowserVoiceBridge` (e.g. after `_handle_barge_in`):

```python
    async def _emit_outcome(self) -> None:
        """Analyze the finished call and push the outcome to the browser. Idempotent."""
        if self._outcome_emitted or self._llm is None or self._agent is None:
            return
        self._outcome_emitted = True
        try:
            from datetime import datetime, timezone

            transcript = [
                m for m in getattr(self._agent.session, "turns", [])
                if isinstance(m, LLMMessage)
            ]
            analysis = await analyze_call(
                transcript=transcript,
                slots=self._agent.slots.values,
                telephony_status=None,
                final_action=self._last_action,
                tenant_timezone=self._tenant_timezone,
                now=datetime.now(timezone.utc),
                llm=self._llm,
            )
        except Exception:  # noqa: BLE001 - never let analysis break teardown
            log.exception("call outcome analysis failed")
            return
        cb = analysis.callback_datetime
        await self._send_json({
            "type": "outcome",
            "outcome": analysis.outcome.value,
            "summary": analysis.summary,
            "notes": analysis.notes,
            "callback_datetime": cb.isoformat() if cb else None,
            "callback_phrase": analysis.callback_phrase,
            "source": analysis.analysis_source,
        })
```

In `run()`'s `finally` block, before `await self._agent.handle_hangup()`, add:

```python
            try:
                await self._emit_outcome()
            except Exception:  # noqa: BLE001
                log.exception("emit outcome failed")
```

Wire the factory in `src/api/dev_console.py` (the `factory()` that builds `BrowserVoiceBridge`, ~line 144) to pass the LLM and tenant timezone — it already has `llm = providers.get_llm(tenant)` and `tenant` in scope. Add to the `BrowserVoiceBridge(...)` constructor call:

```python
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_browser_bridge.py -v`
Expected: PASS (existing tests + the new one).

- [ ] **Step 6: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge.py
git commit -m "feat(outcome): dev console emits call outcome on end"
```

---

## Task 6: Dev console — outcome results panel

**Files:**
- Modify: `static/dev_console.html`

No automated test (static HTML/JS); verified manually via the running dev console.

- [ ] **Step 1: Add the panel markup**

In `static/dev_console.html`, after the `#state` element (the slots/state panel), add:

```html
    <div id="outcome" style="margin-top:12px; padding:8px; border:1px solid #333; border-radius:6px; display:none;">
      <div><b>Outcome:</b> <span id="outcomeType"></span> <span id="outcomeSource" style="color:#888;"></span></div>
      <div style="margin-top:4px;"><b>Summary:</b> <span id="outcomeSummary"></span></div>
      <div style="margin-top:4px;"><b>Notes:</b> <span id="outcomeNotes"></span></div>
      <div id="outcomeCallback" style="margin-top:4px;"></div>
    </div>
```

- [ ] **Step 2: Handle the `outcome` message**

In `static/dev_console.html`, in the `ws.onmessage` handler, add a branch alongside the other `msg.type` checks (e.g. after the `state` branch):

```javascript
    } else if (msg.type === "outcome") {
      $("outcome").style.display = "block";
      $("outcomeType").textContent = msg.outcome;
      $("outcomeSource").textContent = msg.source ? "(" + msg.source + ")" : "";
      $("outcomeSummary").textContent = msg.summary || "—";
      $("outcomeNotes").textContent = msg.notes || "—";
      $("outcomeCallback").textContent = msg.callback_datetime
        ? "Callback: " + msg.callback_datetime
        : (msg.callback_phrase ? "Callback (unresolved): " + msg.callback_phrase : "");
```

- [ ] **Step 3: Verify manually**

```bash
# server already runs on :8765 (restart if needed):
# VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env
```
Open http://localhost:8765/dev/voice, complete a short call, end it (close the tab's mic / let it reach a terminal turn), and confirm the outcome panel populates with type + summary + notes.

- [ ] **Step 4: Commit**

```bash
git add static/dev_console.html
git commit -m "feat(outcome): dev console outcome results panel"
```

---

## Task 7: Campaign — enrich CallResult + DB columns

**Files:**
- Modify: `src/campaign/models.py` (`CallResult`)
- Modify: `src/models/conversation.py` (`Conversation`)
- Test: `tests/unit/test_campaign_models.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_campaign_models.py`:

```python
def test_call_result_carries_outcome_and_summary():
    from datetime import datetime
    from src.campaign.models import (
        CallResult, CallDisposition, LeadCallOutcome, disposition_from_outcome,
    )

    outcome = LeadCallOutcome.CALLBACK_REQUESTED
    r = CallResult(
        session_id="s1", tenant_id="t1", campaign_id="c1", lead_id="l1",
        disposition=disposition_from_outcome(outcome),
        outcome=outcome, summary="Wants a callback.", notes="Tomorrow eve.",
        started_at=datetime(2026, 6, 5, 12, 0), ended_at=datetime(2026, 6, 5, 12, 3),
    )
    assert r.outcome == LeadCallOutcome.CALLBACK_REQUESTED
    assert r.disposition == CallDisposition.INTERESTED_CALLBACK
    assert r.summary == "Wants a callback."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_campaign_models.py::test_call_result_carries_outcome_and_summary -v`
Expected: FAIL — `CallResult` has no `outcome`/`summary` fields.

- [ ] **Step 3: Extend `CallResult`**

In `src/campaign/models.py`, add these fields to `class CallResult(BaseModel)` (after the existing `slots` field):

```python
    outcome: Optional[LeadCallOutcome] = None
    summary: str = ""
    notes: str = ""
    callback_datetime: Optional[datetime] = None
```

- [ ] **Step 4: Add DB columns**

In `src/models/conversation.py`, inside `class Conversation(Base)`, after the `disposition` column, add:

```python
    outcome: Mapped[Optional[str]] = mapped_column(String(30))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    callback_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
```

(`String`, `Text`, `DateTime`, `Optional`, `datetime`, `Mapped`, `mapped_column` are already imported in this file.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_campaign_models.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/campaign/models.py src/models/conversation.py tests/unit/test_campaign_models.py
git commit -m "feat(outcome): enrich CallResult + Conversation with outcome/summary/notes/callback"
```

---

## Task 8: Full suite + finish

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (prior 629 + new tests). Investigate any failure before proceeding.

- [ ] **Step 2: Restart the dev server on the new code (if validating live)**

```bash
# kill the old uvicorn, then:
VOX_DEV_CONSOLE=1 nohup .venv/bin/python3 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env >> /private/tmp/voxserver.log 2>&1 &
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/health   # expect 200
```

- [ ] **Step 3: Finish the branch**

Use `superpowers:finishing-a-development-branch` to decide merge/PR. Branch: `feature/call-outcome-analysis`.

---

## Self-Review notes (author)

- **Spec coverage:** taxonomy+mapping (T1), CallAnalysis+tenant tz (T2), LLM pass+callback resolution (T3), telephony short-circuit+fallback (T4), dev console emit (T5) + UI (T6), campaign CallResult+DB (T7). All spec sections covered.
- **Type consistency:** `LeadCallOutcome`, `CallAnalysis`, `disposition_from_outcome`, `outcome_from_telephony`, `analyze_call(**kwargs)` signature used identically across tasks.
- **Resolved against the codebase:** engine LLM attr is `self._llm` (engine.py:204); the bridge gets `llm` + `tenant_timezone` from the dev_console factory (which has both) instead of reaching through the agent; bridge tests use `_bridge(ws, agent)` + `FakeWebSocket.sent_text` + `FakeAgent` (`session.turns`, `slots.values`). `CallDisposition` values confirmed: `interested_transfer`, `interested_callback`, `not_interested`, `dnd_requested`, `busy_retry`, `voicemail`.
