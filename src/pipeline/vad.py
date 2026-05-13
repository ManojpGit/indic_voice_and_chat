"""Voice Activity Detection.

Two implementations behind a common ``VADDetector`` Protocol:

1. ``EnergyVAD`` — a 30-line RMS-threshold detector. No model dependencies,
   useful for tests and as a fallback. Quality is rough but predictable.

2. ``SileroVAD`` — wraps the ``snakers4/silero-vad`` ONNX model via the
   ``silero-vad`` PyPI package. Imports lazily so users who don't need it
   (i.e. anyone running unit tests) don't pay the install cost.

Both consume 16-bit mono PCM and report per-frame ``is_speech``. The
pipeline engine uses the detector to find utterance boundaries (endpointing)
and to drive the interruption handler (barge-in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from src.pipeline.audio_utils import rms_energy_pcm16


@dataclass
class VADFrame:
    is_speech: bool
    energy: float
    probability: float = 0.0  # filled by SileroVAD; EnergyVAD leaves at 0.0


class VADDetector(Protocol):
    sample_rate: int
    frame_ms: int

    def detect(self, pcm16: bytes) -> VADFrame: ...

    def reset(self) -> None: ...


# --- EnergyVAD ----------------------------------------------------------


class EnergyVAD:
    """Trivial RMS-threshold VAD. Noisy environments will need Silero."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        rms_threshold: float = 300.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._threshold = rms_threshold

    @property
    def frame_bytes(self) -> int:
        # 16-bit mono => 2 bytes per sample
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    def detect(self, pcm16: bytes) -> VADFrame:
        energy = rms_energy_pcm16(pcm16)
        return VADFrame(is_speech=energy >= self._threshold, energy=energy)

    def reset(self) -> None:
        pass


# --- SileroVAD ----------------------------------------------------------


class SileroVAD:
    """Silero VAD wrapper. Loads the ONNX model lazily on first use."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        threshold: float = 0.5,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._threshold = threshold
        self._model = None  # lazy
        self._utils = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # Lazy import — pulls torch in. Tests should not exercise this path.
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "SileroVAD requires the 'silero-vad' package and torch. "
                "Install with: pip install silero-vad torch"
            ) from e
        self._model = load_silero_vad(onnx=True)

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    def detect(self, pcm16: bytes) -> VADFrame:
        self._ensure_model()
        import numpy as np
        import torch  # type: ignore[import-not-found]

        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return VADFrame(is_speech=False, energy=0.0, probability=0.0)
        tensor = torch.from_numpy(samples)
        prob = float(self._model(tensor, self.sample_rate).item())
        return VADFrame(
            is_speech=prob >= self._threshold,
            energy=rms_energy_pcm16(pcm16),
            probability=prob,
        )

    def reset(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()


# --- Endpoint detection -------------------------------------------------


@dataclass
class EndpointConfig:
    min_speech_ms: int = 250
    min_silence_ms: int = 600


class EndpointDetector:
    """Tracks speech/silence runs to detect end-of-utterance.

    Feed VAD frames in order; ``utterance_complete()`` returns True once
    we've seen at least ``min_speech_ms`` of speech followed by
    ``min_silence_ms`` of contiguous silence.
    """

    def __init__(self, frame_ms: int, cfg: EndpointConfig) -> None:
        self._frame_ms = frame_ms
        self._cfg = cfg
        self._speech_ms = 0
        self._trailing_silence_ms = 0
        self._saw_enough_speech = False

    def feed(self, frame: VADFrame) -> bool:
        if frame.is_speech:
            self._speech_ms += self._frame_ms
            self._trailing_silence_ms = 0
            if self._speech_ms >= self._cfg.min_speech_ms:
                self._saw_enough_speech = True
            return False
        if self._saw_enough_speech:
            self._trailing_silence_ms += self._frame_ms
            return self._trailing_silence_ms >= self._cfg.min_silence_ms
        return False

    def feed_many(self, frames: Iterable[VADFrame]) -> bool:
        complete = False
        for f in frames:
            if self.feed(f):
                complete = True
                break
        return complete

    def reset(self) -> None:
        self._speech_ms = 0
        self._trailing_silence_ms = 0
        self._saw_enough_speech = False
