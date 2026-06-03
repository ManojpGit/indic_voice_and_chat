"""Streaming voice pipeline engine.

Coordinates STT -> LLM -> TTS for a single conversational turn:

    captured_audio (bytes)
        |
        v
    STT.transcribe   -> user_text (with confidence + language)
        |
        v
    LLM.generate_stream -> token stream
        |
        v       (split on sentence boundaries via SentenceDetector)
        v
    TTS.synthesize_stream -> audio chunks
        |
        v
    audio_sink (caller-provided callable)

Design choices:
- Stages overlap: as soon as the LLM emits one complete sentence, we kick
  off TTS on it while the LLM keeps generating the next sentence.
- The full LLM text is also returned at the end so the caller can parse the
  structured JSON response (the streamed audio is just the speakable part).
- Per-stage latency is recorded in ``TurnMetrics`` for benchmarking.
- Cancellable via the supplied ``asyncio.Event`` (set by interruption
  handler to drop in-flight audio).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage
from src.interfaces.stt import ISTTProvider, STTConfig
from src.interfaces.tts import ITTSProvider, TTSConfig
from src.pipeline.sentence_detector import SentenceDetector


AudioSink = Callable[[bytes], Awaitable[None]]


def _speakable_from_json(raw: str) -> str:
    """Extract the spoken text (the ``response_text`` field) from a structured
    JSON LLM response.

    When the LLM runs in ``response_format=json`` mode it emits an envelope
    like ``{"response_text": "...", "action": "...", "updated_slots": {...}}``.
    Only ``response_text`` should be spoken — feeding the raw envelope to TTS
    makes it read field names ("response_text" -> "response underscore text"),
    braces, and slot keys aloud. Tolerant of markdown code fences and
    surrounding prose; returns '' when no ``response_text`` can be recovered.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001 - tolerant: fall back to a {...} search
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                obj = None
    if isinstance(obj, dict):
        return str(obj.get("response_text") or "")
    return ""


@dataclass
class TurnMetrics:
    stt_latency_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_first_chunk_ms: int = 0
    tts_total_ms: int = 0
    total_latency_ms: int = 0


@dataclass
class TurnResult:
    user_text: str
    user_language: Optional[str]
    user_confidence: float
    agent_text: str  # full raw LLM output (for parsing)
    audio_bytes_sent: int
    metrics: TurnMetrics
    cancelled: bool = False
    sentences_spoken: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig


class PipelineEngine:
    """One-call-per-instance is fine; reuse across calls is also OK since
    state is held only in local variables of ``run_turn``.
    """

    def __init__(
        self,
        stt: ISTTProvider,
        llm: ILLMProvider,
        tts: ITTSProvider,
        config: PipelineConfig,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._config = config

    async def run_turn(
        self,
        captured_audio: bytes,
        history: list[LLMMessage],
        audio_sink: AudioSink,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> TurnResult:
        """Run one perception-reasoning-action cycle.

        ``history`` is the full list of LLMMessages including the system
        prompt and prior turns. The caller is responsible for appending
        the new user turn before calling ``run_turn``... or not, it's fine
        either way: ``run_turn`` does NOT mutate ``history``.
        """
        cancel_event = cancel_event or asyncio.Event()
        metrics = TurnMetrics()
        t_overall = time.perf_counter()

        # --- STT ---------------------------------------------------------
        t0 = time.perf_counter()
        stt_result = await self._stt.transcribe(captured_audio, self._config.stt)
        metrics.stt_latency_ms = int((time.perf_counter() - t0) * 1000)

        # If STT returned nothing useful, exit early — caller decides what
        # to do (re-prompt, end the call, etc.).
        if not stt_result.text.strip():
            metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)
            return TurnResult(
                user_text="",
                user_language=stt_result.language,
                user_confidence=stt_result.confidence,
                agent_text="",
                audio_bytes_sent=0,
                metrics=metrics,
            )

        # Build the messages list for the LLM.
        messages = list(history) + [
            LLMMessage(role="user", content=stt_result.text)
        ]

        # --- LLM streaming + TTS streaming (overlapped) ------------------
        detector = SentenceDetector()
        full_text_parts: list[str] = []
        sentences_spoken: list[str] = []
        bytes_sent = 0
        first_token_at: Optional[float] = None
        first_audio_at: Optional[float] = None

        t_llm_start = time.perf_counter()

        # Queue of completed sentences awaiting TTS.
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def tts_worker() -> None:
            """Drain sentence_queue, synthesize each, push audio to sink."""
            nonlocal first_audio_at, bytes_sent
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await self._tts.synthesize(sentence, self._config.tts)
                except Exception:
                    # TTS failures shouldn't kill the turn; log via raise
                    # in the caller. Best-effort: swallow and move on.
                    continue
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)
                await audio_sink(result.audio)

        tts_task = asyncio.create_task(tts_worker())

        # In JSON mode the stream is a structured envelope, not speech: buffer
        # it and speak only the parsed response_text. In plain-text mode we
        # stream tokens to TTS sentence-by-sentence for low latency.
        is_json = getattr(self._config.llm, "response_format", None) == "json"

        try:
            async for token in self._llm.generate_stream(messages, self._config.llm):
                if cancel_event.is_set():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                full_text_parts.append(token)
                if not is_json:
                    for sentence in detector.feed(token):
                        await sentence_queue.put(sentence)

            # Flush the LLM tail. Only speak it if cancellation hasn't fired.
            if not cancel_event.is_set():
                if is_json:
                    # Speak only response_text, never the raw JSON envelope.
                    for sentence in detector.feed(_speakable_from_json("".join(full_text_parts))):
                        await sentence_queue.put(sentence)
                for sentence in detector.flush():
                    await sentence_queue.put(sentence)
        finally:
            await sentence_queue.put(None)
            await tts_task

        metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
        if first_token_at is not None:
            metrics.llm_ttft_ms = int((first_token_at - t_llm_start) * 1000)
        if first_audio_at is not None:
            metrics.tts_first_chunk_ms = int((first_audio_at - t_llm_start) * 1000)
            metrics.tts_total_ms = int((time.perf_counter() - first_audio_at) * 1000)
        metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)

        return TurnResult(
            user_text=stt_result.text,
            user_language=stt_result.language,
            user_confidence=stt_result.confidence,
            agent_text="".join(full_text_parts),
            audio_bytes_sent=bytes_sent,
            metrics=metrics,
            cancelled=cancel_event.is_set(),
            sentences_spoken=sentences_spoken,
        )
