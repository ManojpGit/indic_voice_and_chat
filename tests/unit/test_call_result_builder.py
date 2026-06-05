from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.analysis.call_outcome import analyze_agent_call, build_call_result
from src.campaign.models import CallDisposition, LeadCallOutcome
from src.interfaces.llm import LLMMessage, LLMResult


class FakeLLM:
    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    async def generate(self, messages, config) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, finish_reason="stop")

    async def generate_stream(self, messages, config):  # pragma: no cover
        raise NotImplementedError


CALLBACK_JSON = (
    '{"outcome": "callback_requested", "summary": "Wants a callback tomorrow.", '
    '"notes": "Busy now.", "callback_datetime": "2026-06-06T17:00:00", '
    '"callback_phrase": "kal shaam"}'
)
NOW = datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
TRANSCRIPT = [LLMMessage(role="user", content="kal shaam call karna")]


@pytest.mark.asyncio
async def test_build_call_result_populates_and_maps_disposition():
    result = await build_call_result(
        session_id="s1", tenant_id="t1", campaign_id="c1", lead_id="l1",
        transcript=TRANSCRIPT, slots={"interest_level": "warm"},
        telephony_status=None, final_action="schedule_callback",
        tenant_timezone="Asia/Kolkata", now=NOW, llm=FakeLLM(CALLBACK_JSON),
        started_at=datetime(2026, 6, 5, 12, 0), ended_at=datetime(2026, 6, 5, 12, 4),
        duration_ms=240000, total_turns=6, sentiment_history=["neutral", "positive"],
        interest_level="warm",
    )
    assert result.outcome == LeadCallOutcome.CALLBACK_REQUESTED
    assert result.disposition == CallDisposition.INTERESTED_CALLBACK  # mapped
    assert result.summary == "Wants a callback tomorrow."
    assert result.callback_datetime == datetime(2026, 6, 6, 17, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert result.session_id == "s1" and result.lead_id == "l1"
    assert result.duration_ms == 240000 and result.total_turns == 6
    assert result.interest_level == "warm"
    assert result.slots == {"interest_level": "warm"}


@pytest.mark.asyncio
async def test_build_call_result_telephony_short_circuit_no_llm_call():
    llm = FakeLLM("{}")
    result = await build_call_result(
        session_id="s2", tenant_id="t1", campaign_id="c1", lead_id="l2",
        transcript=[], slots={}, telephony_status="busy", final_action=None,
        tenant_timezone="Asia/Kolkata", now=NOW, llm=llm,
        started_at=NOW, ended_at=NOW,
    )
    assert result.outcome == LeadCallOutcome.BUSY
    assert result.disposition == CallDisposition.BUSY_RETRY
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_analyze_agent_call_pulls_transcript_from_agent():
    agent = SimpleNamespace(
        session=SimpleNamespace(turns=[LLMMessage(role="user", content="haan")]),
        slots=SimpleNamespace(values={"x": 1}),
    )
    analysis = await analyze_agent_call(
        agent, llm=FakeLLM('{"outcome":"interested","summary":"ok","notes":"n"}'),
        tenant_timezone="Asia/Kolkata", final_action="close_positive", now=NOW,
    )
    assert analysis is not None
    assert analysis.outcome == LeadCallOutcome.INTERESTED


@pytest.mark.asyncio
async def test_analyze_agent_call_returns_none_without_llm():
    agent = SimpleNamespace(session=SimpleNamespace(turns=[]), slots=SimpleNamespace(values={}))
    assert await analyze_agent_call(
        agent, llm=None, tenant_timezone="Asia/Kolkata", final_action=None, now=NOW
    ) is None


@pytest.mark.asyncio
async def test_analyze_agent_call_defensive_when_agent_lacks_session():
    agent = SimpleNamespace()  # no session/slots, like the telephony FakeAgent
    analysis = await analyze_agent_call(
        agent, llm=FakeLLM('{"outcome":"not_interested","summary":"s","notes":"n"}'),
        tenant_timezone="Asia/Kolkata", final_action=None, now=NOW,
    )
    assert analysis is not None
    assert analysis.outcome == LeadCallOutcome.NOT_INTERESTED
