"""Text-to-speech synthesis for BT, using Piper.

Model loading and synthesis are both CPU-bound, blocking calls, and
synthesize() returns a lazy generator of audio chunks — same pattern as
bt_core/stt/transcriber.py: the call and full generator consumption run
together inside asyncio.to_thread, so the real work never touches the
event loop and iteration doesn't leak blocking work back onto it.
"""

from __future__ import annotations

import asyncio

import numpy as np
from piper import PiperVoice

from bt_core.config import TtsConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)


class Synthesizer:
    """Async wrapper around a Piper voice for text-to-speech."""

    def __init__(self, config: TtsConfig) -> None:
        """Initialize the synthesizer. Model loading happens in :meth:`load`.

        Args:
            config: TTS section of BT's settings (voice model path, sample
                rate).
        """
        self._config = config
        self._voice: PiperVoice | None = None

    async def load(self) -> None:
        """Load the Piper voice model."""
        log.info("tts_model_loading", voice_model_path=str(self._config.voice_model_path))
        self._voice = await asyncio.to_thread(PiperVoice.load, self._config.voice_model_path)
        log.info("tts_model_loaded", voice_model_path=str(self._config.voice_model_path))

    async def synthesize(self, text: str) -> np.ndarray:
        """Convert text to a float32 audio buffer.

        Args:
            text: The text to speak.

        Returns:
            A ``(n_samples,)`` float32 mono array at ``config.sample_rate``.

        Raises:
            RuntimeError: If called before :meth:`load`.
        """
        if self._voice is None:
            raise RuntimeError("Synthesizer.load() must be called before synthesize()")

        audio = await asyncio.to_thread(self._synthesize_blocking, text)
        log.info("tts_synthesized", text_length=len(text), audio_samples=len(audio))
        return audio

    def _synthesize_blocking(self, text: str) -> np.ndarray:
        """Run synthesize() and fully consume its lazy chunk generator.

        Must only be called from a worker thread (via asyncio.to_thread) —
        this is where the actual blocking synthesis work happens.
        """
        assert self._voice is not None
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)
