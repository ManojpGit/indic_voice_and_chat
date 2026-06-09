from src.api.telephony_stringee import (
    answer_scco,
    closing_scco,
    reply_scco,
    reprompt_scco,
)


def test_answer_scco_plays_opening_then_records():
    scco = answer_scco(audio_url="https://x/a.wav", event_url="https://x/ev")
    assert scco[0]["action"] == "play"
    assert scco[0]["url"] == "https://x/a.wav"
    assert scco[0]["bargeIn"] is True
    rec = scco[1]
    assert rec["action"] == "recordMessage"
    assert rec["eventUrl"] == "https://x/ev"
    assert rec["format"] == "wav"
    assert rec["silenceTimeout"] == 1500


def test_reply_scco_same_shape():
    scco = reply_scco(audio_url="https://x/r.wav", event_url="https://x/ev")
    assert [a["action"] for a in scco] == ["play", "recordMessage"]
    assert scco[0]["bargeIn"] is True


def test_reprompt_scco_uses_talk_and_records_again():
    scco = reprompt_scco(text="Dobara boliye?", event_url="https://x/ev")
    assert scco[0]["action"] == "talk"
    assert scco[0]["text"] == "Dobara boliye?"
    assert scco[1]["action"] == "recordMessage"


def test_closing_scco_plays_then_hangs_up():
    scco = closing_scco(audio_url="https://x/c.wav")
    assert [a["action"] for a in scco] == ["play", "hangup"]
    assert "eventUrl" not in scco[0]
