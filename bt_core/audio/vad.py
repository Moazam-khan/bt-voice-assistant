"""Voice activity detection for BT, using Silero VAD.

Silero's model requires exactly 512 samples per call at 16kHz (32ms).
Audio capture's block size is independently configurable, so this module
buffers incoming chunks internally and only invokes the model once enough
samples have accumulated — callers can feed chunks of any size.

Loads the ONNX build of the model (onnx=True), not the default
TorchScript/.jit build: the .jit variant hung indefinitely on first load
inside a PyInstaller-packaged build (a known category of issue with
torch.jit.load() needing things a frozen bundle doesn't have), while the
onnxruntime-based path — already proven working here via the wake word
model — loads reliably either way. VADIterator supports both formats
interchangeably.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from bt_core.config import VadConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_SILERO_CHUNK_SAMPLES = 512  # fixed requirement of Silero VAD v5+ at 16kHz
_SILERO_SAMPLE_RATE = 16000


class SpeechEvent(str, Enum):
    """A change in speech activity state."""

    START = "start"
    END = "end"


class VoiceActivityDetector:
    """Streaming voice activity detector wrapping Silero VAD.

    Buffers arbitrary-sized incoming audio chunks into the fixed 512-sample
    windows Silero requires, and reports speech start/end as they occur.
    """

    def __init__(self, config: VadConfig, sample_rate: int) -> None:
        """Initialize the detector.

        Args:
            config: VAD section of BT's settings (threshold, silence
                duration).
            sample_rate: Sample rate of incoming audio, in Hz. Must be
                16000 Hz — the only rate this pipeline supports.

        Raises:
            ValueError: If ``sample_rate`` is not 16000.
        """
        if sample_rate != _SILERO_SAMPLE_RATE:
            raise ValueError(
                f"VoiceActivityDetector requires {_SILERO_SAMPLE_RATE}Hz audio, "
                f"got {sample_rate}Hz"
            )
        self._model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            self._model,
            threshold=config.threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=config.min_silence_ms,
        )
        self._buffer: np.ndarray = np.empty((0,), dtype=np.float32)
        log.info(
            "vad_initialized",
            threshold=config.threshold,
            min_speech_ms=config.min_speech_ms,
            min_silence_ms=config.min_silence_ms,
        )

    def reset(self) -> None:
        """Reset internal detector state, e.g. after an utterance completes."""
        self._iterator.reset_states()
        self._buffer = np.empty((0,), dtype=np.float32)

    def process(self, chunk: np.ndarray) -> list[SpeechEvent]:
        """Feed an audio chunk and return any speech events it triggers.

        Args:
            chunk: A ``(n_samples,)`` or ``(n_samples, channels)`` float32
                array. Multi-channel audio is downmixed to mono by averaging.

        Returns:
            Zero or more ``SpeechEvent`` values, in order. Buffering means
            one chunk can produce zero, one, or several events.
        """
        mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk
        self._buffer = np.concatenate([self._buffer, mono.astype(np.float32)])

        events: list[SpeechEvent] = []
        while len(self._buffer) >= _SILERO_CHUNK_SAMPLES:
            window = self._buffer[:_SILERO_CHUNK_SAMPLES]
            self._buffer = self._buffer[_SILERO_CHUNK_SAMPLES:]
            result = self._iterator(torch.from_numpy(window), return_seconds=False)
            if result is None:
                continue
            if "start" in result:
                events.append(SpeechEvent.START)
                log.debug("vad_speech_start")
            if "end" in result:
                events.append(SpeechEvent.END)
                log.debug("vad_speech_end")
        return events
