"""Language-aware sentence boundary detection for streaming TTS.

Used by the pipeline engine to flush LLM output to TTS one sentence at a
time — this is what gives the perception of low latency: the user hears the
first sentence while the LLM is still generating the rest.

Handles:
- English / Latin-script: ``.``, ``!``, ``?``, ``;``
- Devanagari (Hindi, Marathi, Sanskrit): ``।`` (purna viram U+0964)
- Quoted speech and ellipses

Word-internal periods (``Dr.``, ``Mrs.``) and decimals (``3.14``) are NOT
sentence ends.
"""

from __future__ import annotations

import re

# Punctuation that can end a sentence in any of our supported scripts.
_TERMINATORS = ".!?;।॥"

# Common abbreviations whose trailing period is NOT a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "i.e", "e.g", "etc", "vs", "no", "fig",
    "rs", "inc", "ltd", "co",
}


class SentenceDetector:
    """Stateful detector — feed text fragments, flush completed sentences."""

    def __init__(self, min_chars: int = 4) -> None:
        # Below ``min_chars`` we don't emit even at a terminator. Stops a
        # stray "Hi." from being flushed before the rest of the sentence.
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Append ``text`` to the buffer, return any sentences now complete."""
        if not text:
            return []
        self._buffer += text
        return self._drain()

    def flush(self) -> list[str]:
        """Force-emit any pending buffered text as a final sentence."""
        out: list[str] = []
        rest = self._buffer.strip()
        if rest:
            out.append(rest)
        self._buffer = ""
        return out

    def reset(self) -> None:
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    # --- internals -----------------------------------------------------

    def _drain(self) -> list[str]:
        emitted: list[str] = []
        # Index from which to keep scanning the buffer for the next terminator.
        scan_start = 0
        while scan_start < len(self._buffer):
            idx = self._next_terminator_index(self._buffer, scan_start)
            if idx == -1:
                break
            # Pull in any trailing closing quotes / brackets so they go with
            # this sentence rather than starting the next one.
            end = idx + 1
            while end < len(self._buffer) and self._buffer[end] in '"\'”’)]':
                end += 1

            candidate = self._buffer[:end].strip()
            if len(candidate) < self._min_chars:
                scan_start = end
                continue
            if not _is_real_sentence_end(self._buffer, idx):
                scan_start = end
                continue
            emitted.append(candidate)
            self._buffer = self._buffer[end:].lstrip()
            scan_start = 0
        return emitted

    @staticmethod
    def _next_terminator_index(buf: str, start: int) -> int:
        for i in range(start, len(buf)):
            if buf[i] in _TERMINATORS:
                return i
        return -1


def _is_real_sentence_end(buf: str, idx: int) -> bool:
    """Is the terminator at ``buf[idx]`` actually a sentence boundary?"""
    ch = buf[idx]
    if ch in "।॥!?;":
        return True
    # ch == "."
    # Decimal: digits on both sides
    left = buf[idx - 1] if idx > 0 else ""
    right = buf[idx + 1] if idx + 1 < len(buf) else ""
    if left.isdigit() and right.isdigit():
        return False
    # Abbreviation: last whitespace-separated token before the period
    prefix = buf[:idx]
    last_token = re.split(r"\s+", prefix)[-1].lower()
    last_token = last_token.rstrip(",;:'\"()")
    if last_token in _ABBREVIATIONS:
        return False
    return True
