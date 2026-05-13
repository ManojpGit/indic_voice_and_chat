from __future__ import annotations

from src.pipeline.sentence_detector import SentenceDetector


def test_emits_on_period() -> None:
    d = SentenceDetector()
    out = d.feed("Hello world. ")
    assert out == ["Hello world."]


def test_emits_on_question_mark() -> None:
    d = SentenceDetector()
    out = d.feed("Aap kaise hain?")
    assert out == ["Aap kaise hain?"]


def test_emits_on_devanagari_purna_viram() -> None:
    d = SentenceDetector()
    out = d.feed("नमस्ते। आप कैसे हैं।")
    assert out == ["नमस्ते।", "आप कैसे हैं।"]


def test_streaming_token_by_token() -> None:
    d = SentenceDetector()
    emitted: list[str] = []
    for tok in ["Hel", "lo ", "world", ". How", " are", " you?"]:
        emitted.extend(d.feed(tok))
    emitted.extend(d.flush())
    assert emitted == ["Hello world.", "How are you?"]


def test_decimal_not_sentence_boundary() -> None:
    d = SentenceDetector()
    out = d.feed("The rate is 3.14 percent. ")
    assert out == ["The rate is 3.14 percent."]


def test_abbreviation_not_sentence_boundary() -> None:
    d = SentenceDetector()
    out = d.feed("Hello Dr. Sharma, kaise hain. ")
    assert out == ["Hello Dr. Sharma, kaise hain."]


def test_short_fragment_not_emitted_alone() -> None:
    d = SentenceDetector(min_chars=4)
    out = d.feed("Hi. World is round.")
    # "Hi." is below min_chars so it gets glued onto the next sentence
    assert out == ["Hi. World is round."]


def test_flush_emits_remaining_unterminated() -> None:
    d = SentenceDetector()
    d.feed("Pending sentence with no terminator")
    assert d.flush() == ["Pending sentence with no terminator"]
    assert d.flush() == []  # idempotent


def test_reset_clears_buffer() -> None:
    d = SentenceDetector()
    d.feed("buffered")
    d.reset()
    assert d.flush() == []


def test_pending_property() -> None:
    d = SentenceDetector()
    d.feed("buffered ")
    assert d.pending == "buffered "


def test_closing_quote_attaches_to_sentence() -> None:
    d = SentenceDetector()
    out = d.feed('She said "Namaste." Then she left.')
    assert any('Namaste.' in s for s in out)
