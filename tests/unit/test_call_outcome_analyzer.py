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
        self.last_config = config
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
async def test_analysis_caps_output_to_avoid_truncated_json():
    # Regression: an unbounded "notes" field let the model overflow max_tokens
    # mid-string, producing invalid (unterminated) JSON -> wrong fallback
    # outcome + empty summary. The prompt must cap output length and the config
    # must give token headroom so a valid envelope always completes.
    llm = FakeLLM('{"outcome": "interested", "summary": "ok", "notes": "n", '
                  '"callback_datetime": null, "callback_phrase": null}')
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="continue", tenant_timezone="Asia/Kolkata", now=now, llm=llm,
    )
    system = llm.last_messages[0].content
    assert "at most 2 short sentences" in system  # explicit notes cap
    assert llm.last_config.max_tokens >= 1536  # headroom beyond the old 1024


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
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, config):
        self.calls += 1
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


@pytest.mark.asyncio
async def test_fenced_json_is_parsed_not_fallback():
    # Gemini sometimes wraps JSON in ```json fences even in JSON mode — the
    # analyzer must still parse it (regression: raw json.loads -> JSONDecodeError
    # -> spurious fallback, seen live).
    fenced = (
        "```json\n"
        '{"outcome": "interested", "summary": "Lead is keen.", "notes": "Send link.", '
        '"callback_datetime": null, "callback_phrase": null}\n'
        "```"
    )
    now = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = await analyze_call(
        transcript=TRANSCRIPT, slots={}, telephony_status=None,
        final_action="close_positive", tenant_timezone="Asia/Kolkata",
        now=now, llm=FakeLLM(fenced),
    )
    assert result.outcome == LeadCallOutcome.INTERESTED
    assert result.analysis_source == "llm"  # NOT "fallback"
    assert result.summary == "Lead is keen."
