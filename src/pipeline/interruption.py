"""Interruption (barge-in) handler.

While the agent is speaking (``RESPONDING`` state), incoming audio from the
caller is fed through ``InterruptionWatcher``. Once speech is detected for
at least ``min_speech_ms`` of contiguous frames, the watcher fires its
callback — the pipeline engine then cancels the in-flight TTS, drops any
pending audio in the playback buffer, and transitions back to LISTENING.

Kept as a small focused class so it's easy to test deterministically with
synthetic frame streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from src.pipeline.vad import VADFrame


@dataclass
class InterruptionConfig:
    min_speech_ms: int = 200       # need this much before declaring barge-in
    detection_interval_ms: int = 20  # nominal frame rate


InterruptCallback = Callable[[], Awaitable[None]]


class InterruptionWatcher:
    def __init__(
        self,
        cfg: InterruptionConfig,
        frame_ms: int,
        on_interrupt: Optional[InterruptCallback] = None,
    ) -> None:
        self._cfg = cfg
        self._frame_ms = frame_ms
        self._on_interrupt = on_interrupt
        self._enabled = False
        self._speech_ms = 0
        self._fired = False

    def enable(self) -> None:
        """Start watching — call when entering RESPONDING state."""
        self._enabled = True
        self._speech_ms = 0
        self._fired = False

    def disable(self) -> None:
        """Stop watching — call when leaving RESPONDING state."""
        self._enabled = False
        self._speech_ms = 0
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    async def feed(self, frame: VADFrame) -> bool:
        """Feed one VAD frame. Returns True if barge-in was detected this call."""
        if not self._enabled or self._fired:
            return False
        if frame.is_speech:
            self._speech_ms += self._frame_ms
            if self._speech_ms >= self._cfg.min_speech_ms:
                self._fired = True
                if self._on_interrupt is not None:
                    await self._on_interrupt()
                return True
        else:
            # Brief silence inside speech shouldn't fully reset; allow up to
            # one frame of jitter before resetting the counter.
            self._speech_ms = max(0, self._speech_ms - self._frame_ms)
        return False
