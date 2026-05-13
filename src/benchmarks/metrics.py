"""Quality + latency metrics used across all benchmarks.

WER and CER use plain Levenshtein edit distance over reference vs hypothesis
(WER on whitespace tokens, CER on Unicode characters). Both are normalized
to ``edits / reference_length`` so a 0.0 score is perfect and 1.0+ means
the hypothesis is at least as long as the reference in pure edits.

For latency we collect raw samples per stage and compute descriptive stats
(mean / median / p95 / p99 / max). No outside deps — these are computed
with stdlib ``statistics`` + manual percentile sort.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Iterable, Optional


# --- text normalization -------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_CHARS = "।॥!?.,;:'\"()[]{}—–-"
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")


def normalize_text(text: str, *, lowercase: bool = True, strip_punct: bool = True) -> str:
    """Cheap text normalization for WER / CER comparisons.

    - NFKC unicode normalize so 'क़' and 'क' + nukta compare equal.
    - Optional lowercase (Latin only — Indic scripts have no case).
    - Optional punctuation strip (including Devanagari ``।``).
    - Collapse whitespace.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    if lowercase:
        out = out.lower()
    if strip_punct:
        out = _PUNCT_RE.sub(" ", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


# --- Edit distance ------------------------------------------------------


def _levenshtein(a: list, b: list) -> int:
    """Generic Levenshtein over two sequences of hashables."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Two-row DP; keep memory at O(min(la, lb)).
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,         # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = cur
    return prev[lb]


@dataclass
class WERResult:
    wer: float
    edits: int
    reference_len: int
    hypothesis_len: int


def word_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> WERResult:
    """WER = edits / reference_words. Empty reference returns wer=0/1 sentinels."""
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    ref_tokens = reference.split() if reference else []
    hyp_tokens = hypothesis.split() if hypothesis else []
    if not ref_tokens:
        # Convention: empty reference but non-empty hypothesis = 1.0; both empty = 0.0
        wer = 0.0 if not hyp_tokens else 1.0
        return WERResult(wer=wer, edits=len(hyp_tokens), reference_len=0, hypothesis_len=len(hyp_tokens))
    edits = _levenshtein(ref_tokens, hyp_tokens)
    return WERResult(
        wer=edits / len(ref_tokens),
        edits=edits,
        reference_len=len(ref_tokens),
        hypothesis_len=len(hyp_tokens),
    )


@dataclass
class CERResult:
    cer: float
    edits: int
    reference_len: int
    hypothesis_len: int


def character_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> CERResult:
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    if not reference:
        cer = 0.0 if not hypothesis else 1.0
        return CERResult(cer=cer, edits=len(hypothesis), reference_len=0, hypothesis_len=len(hypothesis))
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edits = _levenshtein(ref_chars, hyp_chars)
    return CERResult(
        cer=edits / len(ref_chars),
        edits=edits,
        reference_len=len(ref_chars),
        hypothesis_len=len(hyp_chars),
    )


# --- Latency aggregation ------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile. ``q`` is in [0, 100]."""
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    stdev_ms: float = 0.0


def latency_stats(samples_ms: Iterable[float]) -> LatencyStats:
    arr = [float(s) for s in samples_ms]
    if not arr:
        return LatencyStats(
            count=0, mean_ms=0.0, median_ms=0.0,
            p95_ms=0.0, p99_ms=0.0, max_ms=0.0, stdev_ms=0.0,
        )
    return LatencyStats(
        count=len(arr),
        mean_ms=float(mean(arr)),
        median_ms=float(median(arr)),
        p95_ms=percentile(arr, 95.0),
        p99_ms=percentile(arr, 99.0),
        max_ms=float(max(arr)),
        stdev_ms=float(stdev(arr)) if len(arr) >= 2 else 0.0,
    )


# --- Aggregate scoring -------------------------------------------------


@dataclass
class AggregateAccuracy:
    """Aggregated WER + CER over a dataset run."""

    sample_count: int
    wer_mean: float
    cer_mean: float
    wer_p95: float
    cer_p95: float
    per_sample: list[dict] = field(default_factory=list)


def aggregate_accuracy(
    references: list[str],
    hypotheses: list[str],
    *,
    keep_per_sample: bool = False,
) -> AggregateAccuracy:
    """Run WER + CER per row, return aggregated stats."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference / hypothesis length mismatch: {len(references)} vs {len(hypotheses)}"
        )
    wers: list[float] = []
    cers: list[float] = []
    rows: list[dict] = []
    for ref, hyp in zip(references, hypotheses):
        w = word_error_rate(ref, hyp)
        c = character_error_rate(ref, hyp)
        wers.append(w.wer)
        cers.append(c.cer)
        if keep_per_sample:
            rows.append({
                "reference": ref,
                "hypothesis": hyp,
                "wer": w.wer,
                "cer": c.cer,
                "edits_word": w.edits,
                "edits_char": c.edits,
            })
    return AggregateAccuracy(
        sample_count=len(references),
        wer_mean=float(mean(wers)) if wers else 0.0,
        cer_mean=float(mean(cers)) if cers else 0.0,
        wer_p95=percentile(wers, 95.0),
        cer_p95=percentile(cers, 95.0),
        per_sample=rows,
    )
