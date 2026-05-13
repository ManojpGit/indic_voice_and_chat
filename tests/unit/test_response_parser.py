from __future__ import annotations

import json

from src.dialogue.response_parser import (
    parse_chatbot_response,
    parse_voicebot_response,
)


def test_parses_clean_voicebot_json() -> None:
    payload = {
        "response_text": "Namaste!",
        "language": "hi",
        "conversation_phase": "opening",
        "updated_slots": {"lead_name": "Manoj"},
        "action": "continue",
        "action_reason": "greeting",
        "sentiment": "positive",
        "internal_notes": "first turn",
    }
    r = parse_voicebot_response(json.dumps(payload))
    assert r.response_text == "Namaste!"
    assert r.action == "continue"
    assert r.updated_slots == {"lead_name": "Manoj"}
    assert r.parse_error is None


def test_parses_voicebot_inside_markdown_fence() -> None:
    payload = '```json\n{"response_text": "Hi", "language": "en", "action": "continue"}\n```'
    r = parse_voicebot_response(payload)
    assert r.response_text == "Hi"
    assert r.action == "continue"
    assert r.parse_error is None


def test_parses_voicebot_with_leading_text() -> None:
    payload = 'Sure, here\'s the JSON: {"response_text": "Hi", "language": "en", "action": "continue"}'
    r = parse_voicebot_response(payload)
    assert r.response_text == "Hi"


def test_voicebot_unknown_action_falls_back_to_continue() -> None:
    payload = {"response_text": "Hi", "language": "en", "action": "totally_made_up"}
    r = parse_voicebot_response(json.dumps(payload))
    assert r.action == "continue"


def test_voicebot_invalid_phase_dropped() -> None:
    payload = {"response_text": "Hi", "language": "en", "action": "continue", "conversation_phase": "weird"}
    r = parse_voicebot_response(json.dumps(payload))
    assert r.conversation_phase is None


def test_voicebot_missing_response_text_falls_back_to_clarify() -> None:
    payload = {"language": "hi", "action": "continue"}
    r = parse_voicebot_response(json.dumps(payload))
    assert r.action == "clarify"
    assert r.parse_error == "missing response_text"
    assert r.response_text  # non-empty fallback


def test_voicebot_empty_input() -> None:
    r = parse_voicebot_response("")
    assert r.action == "clarify"
    assert r.parse_error == "empty response"


def test_voicebot_garbage_input_salvages_a_speakable_fallback() -> None:
    r = parse_voicebot_response("Hello there, this is not JSON. Just text.")
    assert r.action == "clarify"
    assert "Hello there" in r.response_text


def test_voicebot_truncated_json_takes_largest_balanced() -> None:
    # Outer object incomplete but inner one is fine
    text = '{"response_text": "Hi"} trailing garbage {"x":'
    r = parse_voicebot_response(text)
    assert r.response_text == "Hi"


def test_chatbot_clean_parse() -> None:
    payload = {
        "response_text": "Plan B has 500GB.",
        "language": "en",
        "sources_used": ["plans.pdf:p2"],
        "confidence": "high",
        "action": "none",
        "suggested_followups": ["What's the price?"],
    }
    r = parse_chatbot_response(json.dumps(payload))
    assert r.response_text == "Plan B has 500GB."
    assert r.sources_used == ["plans.pdf:p2"]
    assert r.confidence == "high"


def test_chatbot_invalid_confidence_defaults_medium() -> None:
    payload = {"response_text": "x", "language": "en", "confidence": "extreme"}
    r = parse_chatbot_response(json.dumps(payload))
    assert r.confidence == "medium"


def test_chatbot_empty_input() -> None:
    r = parse_chatbot_response("")
    assert r.parse_error == "empty response"
