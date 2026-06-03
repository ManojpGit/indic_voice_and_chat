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


_SILERO_SESSION = None  # shared, stateless onnxruntime session (state is per-instance)


def _silero_session():
    """Load and cache the bundled silero ONNX model via onnxruntime.

    Uses the model file shipped with the ``silero-vad`` package WITHOUT
    importing the package (which pulls in torch). The session is stateless —
    LSTM state is passed in/out per call — so it is safe to share across VADs.
    """
    global _SILERO_SESSION
    if _SILERO_SESSION is not None:
        return _SILERO_SESSION
    import importlib.util
    import os

    import onnxruntime

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "SileroVAD requires the 'silero-vad' package (for its bundled ONNX "
            "model) and 'onnxruntime'. Install with: pip install silero-vad onnxruntime"
        )
    model_path = os.path.join(os.path.dirname(spec.origin), "data", "silero_vad.onnx")
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    _SILERO_SESSION = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=opts
    )
    return _SILERO_SESSION


class SileroVAD:
    """Silero VAD via onnxruntime (no torch dependency).

    Runs the bundled silero ONNX model directly. It distinguishes speech from
    background noise far better than ``EnergyVAD``, which keeps utterance
    endpointing from running on through room noise / speaker bleed.

    The model requires fixed frame sizes: 512 samples at 16 kHz (``frame_ms=32``)
    or 256 samples at 8 kHz. Feed exactly that many samples per ``detect`` call.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        threshold: float = 0.5,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._threshold = threshold
        self._context_size = 64 if sample_rate == 16000 else 32
        self._session = None  # lazy
        self._state = None
        self._context = None

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        self._session = _silero_session()
        self.reset()

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    def detect(self, pcm16: bytes) -> VADFrame:
        import numpy as np

        self._ensure_model()
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return VADFrame(is_speech=False, energy=0.0, probability=0.0)
        x = samples.reshape(1, -1)
        # silero v5 expects the saved context (last 64 samples) prepended.
        inp = np.concatenate([self._context, x], axis=1).astype(np.float32)
        out, new_state = self._session.run(
            None,
            {
                "input": inp,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = new_state
        self._context = x[:, -self._context_size:]
        prob = float(out[0][0])
        return VADFrame(
            is_speech=prob >= self._threshold,
            energy=rms_energy_pcm16(pcm16),
            probability=prob,
        )

    def reset(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)


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
