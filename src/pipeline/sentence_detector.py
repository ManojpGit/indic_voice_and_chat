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

# First-chunk soft mode (latency): for the FIRST emitted chunk of a turn only,
# we also break on these clause boundaries so TTS can start on a shorter leading
# fragment. The hard terminators above already break; these are the extra ones.
_FIRST_CHUNK_SOFT = ",—–"
# A soft break is only honoured once the leading fragment is at least this many
# chars (so we don't speak a bare "जी," alone — it glues forward instead). Hard
# terminators still use the normal ``min_chars``. Worst case, if no break appears
# within _FIRST_CHUNK_MAX chars we flush at the last word boundary.
_FIRST_CHUNK_MIN = 8
_FIRST_CHUNK_MAX = 40

# Common abbreviations whose trailing period is NOT a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "i.e", "e.g", "etc", "vs", "no", "fig",
    "rs", "inc", "ltd", "co",
}


class SentenceDetector:
    """Stateful detector — feed text fragments, flush completed sentences."""

    def __init__(self, min_chars: int = 4, first_chunk_soft: bool = False) -> None:
        # Below ``min_chars`` we don't emit even at a terminator. Stops a
        # stray "Hi." from being flushed before the rest of the sentence.
        self._min_chars = min_chars
        # When True, the FIRST emitted chunk of this detector's life may break on
        # a soft clause boundary (see _FIRST_CHUNK_SOFT) so TTS starts sooner;
        # after the first emission it reverts to normal full-sentence behaviour.
        self._first_chunk_soft = first_chunk_soft
        self._emitted_any = False
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
        self._emitted_any = False

    @property
    def pending(self) -> str:
        return self._buffer

    # --- internals -----------------------------------------------------

    def _drain(self) -> list[str]:
        emitted: list[str] = []
        # First-chunk soft mode: try to emit a short leading fragment ASAP so the
        # first TTS call starts sooner. Only for the very first emission; if no
        # break point exists yet, emit nothing and wait for more text.
        if self._first_chunk_soft and not self._emitted_any:
            frag = self._take_first_chunk()
            if frag is None:
                return emitted
            emitted.append(frag)
            self._emitted_any = True
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

    def _take_first_chunk(self) -> str | None:
        """Pull a short leading fragment for the FIRST chunk, or None if not ready.

        Breaks on the earliest hard terminator (using the normal ``min_chars``)
        OR soft clause boundary (using the higher _FIRST_CHUNK_MIN, so bare "जी,"
        glues forward) within _FIRST_CHUNK_MAX chars. If neither appears but the
        buffer has reached _FIRST_CHUNK_MAX, flush at the last word boundary so a
        long comma-less opener never blocks first audio (never splits a word)."""
        buf = self._buffer
        limit = min(len(buf), _FIRST_CHUNK_MAX)
        for i in range(limit):
            ch = buf[i]
            is_hard = ch in _TERMINATORS and _is_real_sentence_end(buf, i)
            is_soft = ch in _FIRST_CHUNK_SOFT
            if not (is_hard or is_soft):
                continue
            # Absorb trailing closing quotes/brackets, like the normal path.
            end = i + 1
            while end < len(buf) and buf[end] in '"\'”’)]':
                end += 1
            candidate = buf[:end].strip()
            floor = self._min_chars if is_hard else _FIRST_CHUNK_MIN
            if len(candidate) >= floor:
                self._buffer = buf[end:].lstrip()
                return candidate
        # No usable break within the cap. Once the buffer is long enough, force a
        # flush at the last space so we don't wait on a runaway opener.
        if len(buf) >= _FIRST_CHUNK_MAX:
            cut = buf.rfind(" ", _FIRST_CHUNK_MIN, _FIRST_CHUNK_MAX + 1)
            if cut > 0:
                self._buffer = buf[cut:].lstrip()
                return buf[:cut].strip()
        return None


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
