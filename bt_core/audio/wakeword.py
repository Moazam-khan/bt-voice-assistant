"""Wake word detection for BT, using openWakeWord.

openWakeWord's models expect 16-bit PCM audio (int16), unlike the rest of
this project's audio pipeline, which uses float32 in [-1, 1] throughout —
the conversion happens at this module's boundary so no other module needs
to know about it.
"""

from __future__ import annotations

import numpy as np
from openwakeword.model import Model

from bt_core.config import WakeWordConfig
from bt_core.logging_setup import get_logger

log = get_logger(__name__)


class WakeWordDetector:
    """Detects BT's wake phrase in a stream of audio chunks."""

    def __init__(self, config: WakeWordConfig) -> None:
        """Load the wake word model.

        Args:
            config: Wake word section of BT's settings (model path,
                detection threshold). The melspectrogram/embedding feature
                models openWakeWord also needs are expected alongside the
                wake word model, in the same directory (that's how
                openwakeword.utils.download_models lays them out).
        """
        self._config = config
        models_dir = config.model_path.parent
        self._model = Model(
            wakeword_models=[str(config.model_path)],
            inference_framework="onnx",
            melspec_model_path=str(models_dir / "melspectrogram.onnx"),
            embedding_model_path=str(models_dir / "embedding_model.onnx"),
        )
        self._model_name = next(iter(self._model.models.keys()))
        log.info("wakeword_model_loaded", model=self._model_name, threshold=config.threshold)

    def process(self, chunk: np.ndarray) -> bool:
        """Feed an audio chunk and report whether the wake word just triggered.

        Args:
            chunk: A ``(n_samples,)`` or ``(n_samples, channels)`` float32
                array in [-1, 1], at 16kHz.

        Returns:
            True if the wake word score crossed the configured threshold
            on this chunk.
        """
        mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk
        pcm16 = (mono * 32767).astype(np.int16)
        scores = self._model.predict(pcm16)
        score = scores[self._model_name]
        if score >= self._config.threshold:
            log.info("wakeword_detected", model=self._model_name, score=round(float(score), 3))
            return True
        return False

    def reset(self) -> None:
        """Clear internal audio buffer state, e.g. after a detection is handled."""
        self._model.reset()
