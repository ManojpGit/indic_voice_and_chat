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


# --- first_chunk_soft mode (latency: shorter first chunk) --------------------

def test_first_chunk_breaks_on_comma() -> None:
    d = SentenceDetector(first_chunk_soft=True)
    out = d.feed("देखिए राजू जी, हमारा ऐप official है।")
    # First chunk breaks at the comma (soft boundary); the rest on the danda.
    assert out == ["देखिए राजू जी,", "हमारा ऐप official है।"]


def test_first_chunk_min_prevents_tiny_fragment() -> None:
    d = SentenceDetector(first_chunk_soft=True)
    out = d.feed("जी, हमारा ऐप बहुत अच्छा है।")
    # "जी," is below the soft-break floor, so it glues forward instead of being
    # spoken alone; the whole sentence emits on the danda.
    assert out == ["जी, हमारा ऐप बहुत अच्छा है।"]


def test_first_chunk_max_flush_at_word_boundary() -> None:
    d = SentenceDetector(first_chunk_soft=True)
    text = "this is a fairly long opening clause with no end"
    out = d.feed(text)
    # No punctuation within the cap -> flush at the last word boundary <= max,
    # never mid-word.
    assert out == ["this is a fairly long opening clause"]
    assert len(out[0]) <= 40
    assert text.startswith(out[0])


def test_only_first_chunk_is_soft() -> None:
    d = SentenceDetector(first_chunk_soft=True)
    first = d.feed("देखिए राजू जी, ")
    assert first == ["देखिए राजू जी,"]
    # After the first emission, a later comma must NOT trigger a break.
    assert d.feed("और एक बात, सुनिए ") == []
    assert d.feed("ठीक है।") == ["और एक बात, सुनिए ठीक है।"]


def test_default_unchanged_ignores_commas() -> None:
    # Flag off (default) — commas never break; only terminators do.
    d = SentenceDetector()
    out = d.feed("देखिए राजू जी, हमारा ऐप official है।")
    assert out == ["देखिए राजू जी, हमारा ऐप official है।"]
