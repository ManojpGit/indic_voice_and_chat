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
