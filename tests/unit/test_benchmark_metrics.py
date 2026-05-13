from __future__ import annotations

import pytest

from src.benchmarks.metrics import (
    aggregate_accuracy,
    character_error_rate,
    latency_stats,
    normalize_text,
    percentile,
    word_error_rate,
)


# --- Normalization ------------------------------------------------------


def test_normalize_text_lowercases_and_strips_punct() -> None:
    out = normalize_text("Hello, World! How are you?")
    assert out == "hello world how are you"


def test_normalize_text_handles_devanagari_terminator() -> None:
    out = normalize_text("नमस्ते। आप कैसे हैं।")
    assert "।" not in out
    assert "नमस्ते" in out


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a    b\nc\t\td") == "a b c d"


def test_normalize_text_no_strip_when_disabled() -> None:
    out = normalize_text("Hi!", strip_punct=False)
    assert "!" in out


# --- WER ----------------------------------------------------------------


def test_wer_perfect_match() -> None:
    r = word_error_rate("hello world", "hello world")
    assert r.wer == 0.0
    assert r.edits == 0
    assert r.reference_len == 2


def test_wer_single_substitution() -> None:
    r = word_error_rate("hello world", "hello earth")
    assert r.edits == 1
    assert r.wer == 0.5


def test_wer_insertion() -> None:
    r = word_error_rate("hello world", "hello there world")
    assert r.edits == 1
    assert r.wer == 0.5


def test_wer_deletion() -> None:
    r = word_error_rate("hello there world", "hello world")
    assert r.edits == 1
    assert r.reference_len == 3


def test_wer_handles_empty_reference() -> None:
    assert word_error_rate("", "").wer == 0.0
    assert word_error_rate("", "anything").wer == 1.0


def test_wer_devanagari_match() -> None:
    r = word_error_rate("नमस्ते दोस्त", "नमस्ते दोस्त")
    assert r.wer == 0.0


def test_wer_normalizes_punctuation() -> None:
    r = word_error_rate("Hello, world.", "hello world")
    assert r.wer == 0.0


# --- CER ----------------------------------------------------------------


def test_cer_perfect_match() -> None:
    r = character_error_rate("hello", "hello")
    assert r.cer == 0.0


def test_cer_single_char_diff() -> None:
    r = character_error_rate("hello", "hallo")
    assert r.edits == 1
    assert r.cer == 0.2


def test_cer_empty_reference() -> None:
    assert character_error_rate("", "").cer == 0.0
    assert character_error_rate("", "x").cer == 1.0


# --- Latency stats ------------------------------------------------------


def test_percentile_basic() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 0) == 1


def test_percentile_interpolated() -> None:
    # 95th of 10 values [1..10]: k = 9 * 0.95 = 8.55, between index 8 (9) and 9 (10)
    out = percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95)
    assert 9.0 < out < 10.0


def test_percentile_empty_returns_zero() -> None:
    assert percentile([], 50) == 0.0


def test_latency_stats_full() -> None:
    s = latency_stats([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert s.count == 10
    assert s.mean_ms == 55.0
    assert s.median_ms == 55.0
    assert s.max_ms == 100.0
    assert s.p95_ms > 90.0
    assert s.stdev_ms > 0.0


def test_latency_stats_empty() -> None:
    s = latency_stats([])
    assert s.count == 0
    assert s.mean_ms == 0.0
    assert s.stdev_ms == 0.0


def test_latency_stats_single_sample_zero_stdev() -> None:
    s = latency_stats([42.0])
    assert s.count == 1
    assert s.mean_ms == 42.0
    assert s.stdev_ms == 0.0


# --- Aggregate accuracy -------------------------------------------------


def test_aggregate_accuracy_basic() -> None:
    refs = ["hello world", "good morning"]
    hyps = ["hello world", "good evening"]
    a = aggregate_accuracy(refs, hyps)
    assert a.sample_count == 2
    assert a.wer_mean == pytest.approx(0.25)  # 0 + 0.5 / 2
    assert a.cer_mean > 0.0


def test_aggregate_accuracy_keep_per_sample_off_by_default() -> None:
    a = aggregate_accuracy(["a"], ["a"])
    assert a.per_sample == []


def test_aggregate_accuracy_keep_per_sample_when_requested() -> None:
    a = aggregate_accuracy(["hello"], ["world"], keep_per_sample=True)
    assert len(a.per_sample) == 1
    assert "reference" in a.per_sample[0]
    assert "wer" in a.per_sample[0]


def test_aggregate_accuracy_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_accuracy(["a", "b"], ["a"])
